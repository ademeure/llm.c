/*
Kernels for gelu backward pass.

Compile example:
nvcc -O3 --use_fast_math -lcublas -lcublasLt gelu_backward.cu -o gelu_backward

If encountering "error: identifier "M_PI" is undefined", add the following lines to the top of the file:

#define _USE_MATH_DEFINES
#include <math.h>  OR  #include <cmath>

version 1 is naive port from CPU code to kernel
./gelu_backward 1

version 2 uses the Packed128 data structure
./gelu_backward 2
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define ENABLE_BF16
#include "common.h"

// ----------------------------------------------------------------------------
// CPU code reference

#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)

void gelu_backward_cpu(float* dinp, const float* inp, const float* dout, const int N) {
    for (int i = 0; i < N; i++) {
        float x = inp[i];
        float cube = 0.044715f * x * x * x;
        float tanh_arg = GELU_SCALING_FACTOR * (x + cube);
        float tanh_out = tanhf(tanh_arg);
        float coshf_out = coshf(tanh_arg);
        float sech_out = 1.0f / (coshf_out * coshf_out);
        float local_grad = 0.5f * (1.0f + tanh_out) + x * 0.5f * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x * x);
        dinp[i] = (floatX)(local_grad * (float)dout[i]);
    }
}

// ----------------------------------------------------------------------------
// GPU kernels

// elementwise ops are nice and ez
__global__ void gelu_backward1(floatX* dinp, const floatX* inp, const floatX* dout, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float x = (float)inp[i];
        float cube = 0.044715f * x * x * x;
        float tanh_arg = GELU_SCALING_FACTOR * (x + cube);
        float tanh_out = tanhf(tanh_arg);
        float coshf_out = coshf(tanh_arg);
        float sech_out = 1.0f / (coshf_out * coshf_out);
        float local_grad = 0.5f * (1.0f + tanh_out) + x * 0.5f * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x * x);
        dinp[i] = (floatX)(local_grad * (float)dout[i]);
    }
}

__global__ void gelu_backward2(floatX* dinp, const floatX* inp, const floatX* dout, const int N) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;
    if (i < N) {
        x128 packed_dinp;
        x128 packed_inp = load128cs(inp + i);
        x128 packed_dout = load128cs(dout + i);
        for (int k = 0; k < packed_inp.size; ++k) {
            float x = (float)packed_inp[k];
            float cube = 0.044715f * x * x * x;
            float tanh_arg = GELU_SCALING_FACTOR * (x + cube);
            float tanh_out = tanhf(tanh_arg);
            float coshf_out = coshf(tanh_arg);
            float sech_out = 1.0f / (coshf_out * coshf_out);
            float local_grad = 0.5f * (1.0f + tanh_out) + x * 0.5f * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x * x);
            packed_dinp[k] = (floatX)(local_grad * (float)packed_dout[k]);
        }

        store128(dinp + i, packed_dinp);
    }
}

template <typename Ti, typename Tdout, typename Tdinp>
__global__ void gelu_backward3(Tdinp* dinp, const Ti* inp, const Tdout* dout, const int N) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * Packed128<Tdout>::size;
    if (idx >= N) { return; }

    Packed128<Tdinp> packed_dinp;
    Packed128<Ti> packed_inp = load128cs(inp + idx);
    Packed128<Tdout> packed_dout = load128(dout + idx);
    for (int k = 0; k < Packed128<Tdout>::size; ++k) {
        float x = (float)packed_inp[k];
        float cube = 0.044715f * x * x * x;

        float tanh_in_out = GELU_SCALING_FACTOR * (x + cube);
        #if !defined(PRECISE_GELU_TANH) && __CUDA_ARCH__ >= 750
        asm ("tanh.approx.f32 %0,%1;" : "=f"(tanh_in_out) : "f"(tanh_in_out));
        #else
        tanh_in_out = tanhf(tanh_in_out);
        #endif

        float sech_out = 1.0f - (tanh_in_out * tanh_in_out);
        float local_grad = 0.5f * ((1.0f + tanh_in_out) + x * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x * x));
        float result = local_grad * (float)packed_dout[k];
        packed_dinp[k] = (Tdinp)(result);
    }
    store128(dinp + idx, packed_dinp);
}
// ----------------------------------------------------------------------------
// kernel launcher

void gelu_backward1(floatX* dinp, const floatX* inp, const floatX* dout, int N, const int block_size) {
    const int grid_size = ceil_div(N, block_size);
    gelu_backward1<<<grid_size, block_size>>>(dinp, inp, dout, N);
    cudaCheck(cudaGetLastError());
}

void gelu_backward2(floatX* dinp, const floatX* inp, const floatX* dout, int N, const int block_size) {
    const int grid_size = ceil_div(N, block_size * x128::size);
    gelu_backward2<<<grid_size, block_size>>>(dinp, inp, dout, N);
    cudaCheck(cudaGetLastError());
}

void gelu_backward3(floatX* dinp, const floatX* inp, const floatX* dout, int N, const int block_size) {
    const int grid_size = ceil_div(N, block_size * x128::size);
    gelu_backward3<<<grid_size, block_size>>>(dinp, inp, dout, N);
    cudaCheck(cudaGetLastError());
}

// kernel version dispatch
void gelu_backward(int kernel_num,
                  floatX* dinp,
                  const floatX* inp,
                  const floatX* dout,
                  int B, int T, int C,
                  int block_size) {
    switch (kernel_num) {
        case 1:
            gelu_backward1(dinp, inp, dout, B * T * C, block_size);
            break;
        case 2:
            gelu_backward2(dinp, inp, dout, B * T * C, block_size);
            break;
        case 3:
            gelu_backward3(dinp, inp, dout, B * T * C, block_size);
            break;
        default:
            printf("Invalid kernel number\n");
            exit(1);
    }
}

// ----------------------------------------------------------------------------

int main(int argc, char **argv) {
    setup_main();

    int B = 128;
    int T = 1024;
    int C = 768;

    // create host memory of random numbers
    float* dinp = (float*)malloc(B * T * C * sizeof(float));
    float* inp = make_random_float(B * T * C);
    float* dout = make_random_float(B * T * C);

    // Define the number of BF16 values
    const int num_bf16_values = 65536;

    // Allocate host memory for BF16 input values, output values, and gradient output values
    floatX* bf16_inp = (floatX*)malloc((num_bf16_values/4 * num_bf16_values/4) * sizeof(floatX));
    floatX* bf16_dout = (floatX*)malloc((num_bf16_values/4 * num_bf16_values/4) * sizeof(floatX));
    floatX* bf16_dinp1 = (floatX*)malloc((num_bf16_values/4 * num_bf16_values/4) * sizeof(floatX));
    floatX* bf16_dinp3 = (floatX*)malloc((num_bf16_values/4 * num_bf16_values/4) * sizeof(floatX));

    // Initialize BF16 input values and gradient output values
    int index = 0;
    for (unsigned short i = 0; i < num_bf16_values; i += 1) {
        for (unsigned short j = 0; j < num_bf16_values; j += 99) {
            // reinterpret cast the bits of i and j as bfloat16
            bf16_inp[index] = (floatX)*((__nv_bfloat16*)&i);
            bf16_dout[index] = (floatX)*((__nv_bfloat16*)&j);
            if ((int)j + 99 >= num_bf16_values) {
                break;
            }

            if (fabsf(bf16_inp[index]) >= 1000000.0f || fabsf(bf16_dout[index]) >= 1000000.0f) {
                continue;
            }
            index++;
        }
        if ((int)i + 1 >= num_bf16_values) {
            break;
        }
    }

    // Allocate device memory for BF16 input values, output values, and gradient output values
    floatX* d_bf16_inp;
    floatX* d_bf16_dout;
    floatX* d_bf16_dinp1;
    floatX* d_bf16_dinp3;
    cudaCheck(cudaMalloc(&d_bf16_inp, index * sizeof(floatX)));
    cudaCheck(cudaMalloc(&d_bf16_dout, index * sizeof(floatX)));
    cudaCheck(cudaMalloc(&d_bf16_dinp1, index * sizeof(floatX)));
    cudaCheck(cudaMalloc(&d_bf16_dinp3, index * sizeof(floatX)));

    // Copy BF16 input values and gradient output values to device
    cudaCheck(cudaMemcpy(d_bf16_inp, bf16_inp, index * sizeof(floatX), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_bf16_dout, bf16_dout, index * sizeof(floatX), cudaMemcpyHostToDevice));

    // Run the gelu_backward1 kernel
    gelu_backward(1, d_bf16_dinp1, d_bf16_inp, d_bf16_dout, 1, 1, index, 256);
    cudaCheck(cudaDeviceSynchronize());

    // Run the gelu_backward3 kernel
    gelu_backward(3, d_bf16_dinp3, d_bf16_inp, d_bf16_dout, 1, 1, index, 256);
    cudaCheck(cudaDeviceSynchronize());

    // Copy the results back to host
    cudaCheck(cudaMemcpy(bf16_dinp1, d_bf16_dinp1, index * sizeof(floatX), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(bf16_dinp3, d_bf16_dinp3, index * sizeof(floatX), cudaMemcpyDeviceToHost));

    // Compare the results and print the differences
    float max_diff = 0.0f;

    for (int i = 0; i < index; i++) {
        float diff = fabsf((float)bf16_dinp1[i] - (float)bf16_dinp3[i]);
        if (diff >= 0.00000000000000001f) {
            float percentage = diff / (float)fabsf(bf16_dinp1[i]) * 100.0f;
            printf("[%d]: INPUT %.15f, DOUT %.15f ===> %.15f vs %.15f ===> DIFF: %.15f (%.8f%%)\n",
                    i, (float)bf16_inp[i], (float)bf16_dout[i],
                    (float)bf16_dinp1[i], (float)bf16_dinp3[i], diff, percentage);
        }
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    printf("Maximum difference between gelu_backward1 and gelu_backward3: %.50f\n", max_diff);

    // Free host and device memory
    free(bf16_inp);
    free(bf16_dout);
    free(bf16_dinp1);
    free(bf16_dinp3);
    cudaCheck(cudaFree(d_bf16_inp));
    cudaCheck(cudaFree(d_bf16_dout));
    cudaCheck(cudaFree(d_bf16_dinp1));
    cudaCheck(cudaFree(d_bf16_dinp3));







    // read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);

    // first check the correctness of the kernel
    gelu_backward_cpu(dinp, inp, dout, B * T * C);

    // move to GPU
    floatX* d_dinp;
    floatX* d_inp;
    floatX* d_dout;
    cudaCheck(cudaMalloc(&d_dinp, B * T * C * sizeof(floatX)));
    cudaCheck(cudaMalloc(&d_inp, B * T * C * sizeof(floatX)));
    cudaCheck(cudaMalloc(&d_dout, B * T * C * sizeof(floatX)));

    cudaCheck(memcpy_convert(d_inp, inp, B * T * C));
    cudaCheck(memcpy_convert(d_dout, dout, B * T * C));

    // time the kernel at different block sizes
    int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        printf("Checking block size %d.\n", block_size);
        gelu_backward(kernel_num, d_dinp, d_inp, d_dout, B, T, C, block_size);
#if !defined(ENABLE_BF16) && !defined(ENABLE_FP16)
        float tol = 1e-5f;
#else
        float tol = 1e-3f;
#endif
        validate_result(d_dinp, dinp, "dinp", B * T * C, tol);
    }

    printf("All results match. Starting benchmarks.\n\n");

    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];

        int repeat_times = 1000;

        float elapsed_time = benchmark_kernel(repeat_times, gelu_backward,
                                              kernel_num, d_dinp, d_inp, d_dout,
                                              B, T, C, block_size);

        // napkin math: estimate the memory bandwidth achieved
        // for each (B,T,C) output element, we do 1 read and 1 write, 4 bytes each
        // and e.g. A100 40GB PCIe is advertised at 1,555GB/s
        long memory_ops = B * T * C * 3 * (int)sizeof(floatX);
        float memory_bandwidth = memory_ops / elapsed_time / 1e6;

        printf("block_size %4d | time %.4f ms | bandwidth %.2f GB/s\n", block_size, elapsed_time, memory_bandwidth);
    }

    // free memory
    free(dinp);
    free(inp);
    free(dout);
    cudaCheck(cudaFree(d_dinp));
    cudaCheck(cudaFree(d_inp));
    cudaCheck(cudaFree(d_dout));
    return 0;
}
