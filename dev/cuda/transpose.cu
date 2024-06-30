/*
Kernels for transpose with format conversion

Compile example:
nvcc -O3 --use_fast_math transpose.cu -o transpose

version 0 is a non-optimized copy (not a transpose)
version 1 is an optimized copy (not a transpose)
version
./transpose 1
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda_fp8.h>

#define ENABLE_BF16
#define SKIP_CUBLAS
#include "common.h"

#if !defined(IN_TYPE)
#define IN_TYPE __nv_bfloat16
#endif

#if !defined(OUT_TYPE)
#define OUT_TYPE __nv_fp8_e4m3
#endif

// ----------------------------------------------------------------------------
// CPU code reference

template <typename T1, typename T2>
void transpose_cpu(T1* out, const T2* inp, size_t width, size_t height) {
    for (size_t y = 0; y < height; y++) {
        for (size_t x = 0; x < width; x++) {
            out[x * height + y] = (T1)((IN_TYPE)inp[y * width + x]);
        }
    }
}

// ----------------------------------------------------------------------------
// GPU kernels

template <typename T1, typename T2>
__global__ void copy_kernel(T1 *out, const T2 *inp, size_t N) {
    Packed128<T1> out128;
    size_t n = (blockIdx.x * blockDim.x + threadIdx.x);
    if (n >= N) { return; }
    out[n] = (T1)inp[n];
}

template <typename T1, typename T2>
__global__ void copy128_kernel(T1 *out, const T2 *inp, size_t N) {
    Packed128<T1> out128;
    size_t n = (blockIdx.x * blockDim.x + threadIdx.x) * out128.size;
    if (n >= N) { return; }

    // note: if sizeof(T1) < sizeof(T2), compiler will skip unused elements of load128
    // so it may turn out to be a load32 or load64
    Packed128<T2> inp128 = load128<T2>(inp + n);
    for (int i = 0; i < out128.size; i++) {
        out128[i] = (T1)inp128[i];
    }
    store128<T1>(out + n, out128);
}

template <typename T1, typename T2>
__global__ void transpose_kernel0(T1 *out, const T2 *inp, size_t width, size_t height) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        out[x * height + y] = (T1)inp[y * width + x];
    }
}

#define TILE_DIM 32UL
template<size_t BLOCK_ROWS=8UL, typename T1, typename T2>
__global__ void transpose_kernel1(T1 *odata, const T2 *idata)
{
    __shared__ T1 tile[TILE_DIM][TILE_DIM+1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;
    int height = gridDim.y * TILE_DIM;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        tile[threadIdx.y+j][threadIdx.x] = (T1)idata[(y+j)*width + x];

    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        odata[(y+j)*height + x] = tile[threadIdx.x][threadIdx.y + j];
}

template<size_t BLOCK_ROWS=8UL, typename T1, typename T2>
__global__ void transpose_kernel2(T1 *odata, const T2 *idata)
{
    __shared__ T1 tile[TILE_DIM][TILE_DIM+1];

    int x = blockIdx.x * TILE_DIM + (threadIdx.x * (16 / sizeof(T2)));
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;
    int height = gridDim.y * TILE_DIM;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        Packed128<T2> in128 = load128<T2>(idata + (y+j)*width + x);
        for (int k = 0; k < in128.size; k++) {
            tile[threadIdx.y+j][threadIdx.x * in128.size + k] = (T1)in128[k];
        }
    }

    __syncthreads();

    // this is... not ideal performance-wise, especially for big differences in sizes
    if (threadIdx.x > TILE_DIM / (16 / sizeof(T1))) {
        return;
    }

    x = blockIdx.y * TILE_DIM + (threadIdx.x * (16 / sizeof(T1)));  // transpose block offset
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        Packed128<T1> out128;
        for (int k = 0; k < out128.size; k++) {
            out128[k] = tile[threadIdx.x * out128.size + k][threadIdx.y + j];
        }
        store128<T1>(odata + (y+j)*height + x, out128);
    }
}

// ----------------------------------------------------------------------------
// kernel launcher

template <typename T1, typename T2>
void copy(T1* out, const T2* inp, size_t width, size_t height, const size_t block_size) {
    size_t N = width * height;
    const dim3 grid_size(ceil_div(N, block_size*block_size), 1);
    const dim3 block_size_(block_size*block_size);
    copy_kernel<<<grid_size, block_size_>>>(out, inp, N);
}

template <typename T1, typename T2>
void copy128(T1* out, const T2* inp, size_t width, size_t height, const size_t block_size) {
    size_t N = width * height;
    const dim3 grid_size(ceil_div(N, block_size*block_size * (16 / sizeof(T1))), 1);
    const dim3 block_size_(block_size*block_size);
    copy128_kernel<<<grid_size, block_size_>>>(out, inp, N);
}

template <typename T1, typename T2>
void transpose0(T1* out, const T2* inp, size_t width, size_t height, const size_t block_size) {
    const dim3 grid_size(ceil_div(width, block_size), ceil_div(height, block_size));
    const dim3 block_size_(block_size, block_size);
    transpose_kernel0<<<grid_size, block_size_>>>(out, inp, width, height);
    cudaCheck(cudaGetLastError());
}

template <typename T1, typename T2>
void transpose1(T1* out, const T2* inp, size_t width, size_t height, const size_t block_size) {
    dim3 grid_size(width/(TILE_DIM), height/(TILE_DIM), 1);
    dim3 block_size_(TILE_DIM, block_size, 1);
    if (block_size == 32) {
        transpose_kernel1<32><<<grid_size, block_size_>>>(out, inp);
    } else if (block_size == 16) {
        transpose_kernel1<16><<<grid_size, block_size_>>>(out, inp);
    } else if (block_size == 8) {
        transpose_kernel1<8><<<grid_size, block_size_>>>(out, inp);
    } else if (block_size == 4) {
        transpose_kernel1<4><<<grid_size, block_size_>>>(out, inp);
    } else {
        printf("Invalid block size\n");
        exit(1);
    }
    cudaCheck(cudaGetLastError());
}

template <typename T1, typename T2>
void transpose2(T1* out, const T2* inp, size_t width, size_t height, const size_t block_size) {
    dim3 grid_size(width/(TILE_DIM), height/(TILE_DIM), 1);
    dim3 block_size_(TILE_DIM / (16 / sizeof(T2)), block_size, 1);
    if (block_size == 32) {
        transpose_kernel2<32><<<grid_size, block_size_>>>(out, inp);
    } else if (block_size == 16) {
        transpose_kernel2<16><<<grid_size, block_size_>>>(out, inp);
    } else if (block_size == 8) {
        transpose_kernel2<8><<<grid_size, block_size_>>>(out, inp);
    } else if (block_size == 4) {
        transpose_kernel2<4><<<grid_size, block_size_>>>(out, inp);
    } else {
        printf("Invalid block size\n");
        exit(1);
    }
    cudaCheck(cudaGetLastError());
}

// kernel version dispatch
template <typename T1, typename T2>
void transpose(int kernel_num,
              T1* out, const T2* inp,
              size_t width, size_t height, size_t block_size) {
    switch (kernel_num) {
        case 0:
            copy(out, inp, width, height, block_size);
            break;
        case 1:
            copy128(out, inp, width, height, block_size);
            break;
        case 10:
            transpose0(out, inp, width, height, block_size);
            break;
        case 11:
            transpose1(out, inp, width, height, block_size);
            break;
        case 12:
            transpose2(out, inp, width, height, block_size);
            break;
        default:
            printf("Invalid kernel number\n");
            exit(1);
    }
}

// ----------------------------------------------------------------------------

int main(int argc, const char **argv) {
    setup_main();

    int W = 8192;
    int H = 8192;

    // create host memory of random numbers
    OUT_TYPE* out = (OUT_TYPE*)malloc(W * H * sizeof(OUT_TYPE));
    float* inp = make_random_float_01(W * H);

    // read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);

    // first check the correctness of the kernel
    transpose_cpu(out, inp, W, H);

    // move to GPU
    IN_TYPE *d_inp;
    OUT_TYPE *d_out;
    cudaCheck(cudaMalloc(&d_out, W * H * sizeof(OUT_TYPE)));
    cudaCheck(cudaMalloc(&d_inp, W * H * sizeof(IN_TYPE)));
    cudaCheck(memcpy_convert(d_inp, inp, W * H));

    // time the kernel at different block sizes
    int block_sizes[] = {4, 8, 16, 32};
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        printf("Checking block size %d.\n", block_size);
        transpose(kernel_num, d_out, d_inp, W, H, block_size);
        if (kernel_num >= 10) {
            validate_result(d_out, out, "out", W * H, (OUT_TYPE)1e-5f);
        }
    }
    printf("All results match. Starting benchmarks.\n\n");

    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        int repeat_times = 1000;
        float elapsed_time = benchmark_kernel(repeat_times, transpose<OUT_TYPE, IN_TYPE>,
                                              kernel_num, d_out, d_inp,
                                              W, H, block_size);

        // napkin math: estimate the memory bandwidth achieved
        size_t memory_ops = W * H * (sizeof(IN_TYPE) + sizeof(OUT_TYPE));
        float memory_bandwidth = memory_ops / elapsed_time / 1e6;

        printf("block_size %4d | time %.4f ms | bandwidth %.2f GB/s\n", block_size, elapsed_time, memory_bandwidth);
    }

    free(out);
    free(inp);
    cudaCheck(cudaFree(d_inp));
    cudaCheck(cudaFree(d_out));
}