/*
Kernels for transpose with format conversion
Many parameters are configurable by changing the defines

Compile examples (change 90 to your SM architecture - do not trust performance without it):
nvcc -O3 --generate-code arch=compute_90,code=[compute_90,sm_90] --use_fast_math transpose.cu -o transpose
nvcc -DIN_TYPE=half -DOUT_TYPE=float -DSCALING_FACTOR=0.5f -DTRANSPOSE_AND_COPY=true -O3 --generate-code arch=compute_90,code=[compute_90,sm_90] --use_fast_math transpose.cu -o transpose

version 0 is a non-optimized copy (not a transpose)
version 1 is an optimized copy
version 2 is a different optimized copy

version 10 is a non-optimized transpose
version 11 is an optimized transpose with shared memory
version 12 is a more optimized transpose with shared memory and 128-bit loads/stores

./transpose 12
*/

#define SKIP_CUBLAS
#include "common.h"
#include <cstring>
#include <cuda_fp8.h>

//#define TRANSPOSE_AND_COPY true
#define SCALING_FACTOR 1.5f

#if !defined(TRANSPOSE_AND_COPY)
#define TRANSPOSE_AND_COPY false
#endif

#if !defined(IN_TYPE)
#define IN_TYPE __nv_bfloat16
#endif
#if !defined(OUT_TYPE)
#define OUT_TYPE __nv_fp8_e4m3
#endif

float* d_scaling_factor = NULL;
#if defined(SCALING_FACTOR)
#define SCALING true
#else
#define SCALING_FACTOR 1.0f
#define SCALING false
#endif

#define DEFAULT_TILE 32UL // transpose tile size currently can only be 32x32
#define FIRST_TRANSPOSE_KERNEL 10 // kernels 0/1/2 are copy kernels without transpose

/*
using reduction_func_t = float (*) (float);

template<reduction_func_t warp_reduction>
__device__ inline float blockReduce(float val, bool final_sync, float out_of_bounds) {
    // two reductions of up to 1024 threads:
    // 1) inside warp (shuffle), 2) cross-warp (shared memory), 3) inside warp (shuffle)
    __shared__ float shared_val[WARP_SIZE];
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int num_warps = blockDim.x / WARP_SIZE;

    float warp_val = warp_reduction(val);
    if (lane_id == 0) { shared_val[warp_id] = warp_val; }
    __syncthreads();
    warp_val = (lane_id < num_warps) ? shared_val[lane_id] : out_of_bounds;
    float block_val = warp_reduction(warp_val);

    if (final_sync) {
        __syncthreads(); // only needed in loops when effectively reusing shared memory etc.
    }
    return block_val;
}
*/

// ----------------------------------------------------------------------------
// This helper is for when we want to copy from e.g. FP32 to BF16
// e.g. if want to load a f128 of 4 elements, and write those 4 elements to memory as 64-bit
// not needed in the case of loads, the compiler will automatically optimise away unused reads
// (we might want to replace this with something like a fixed vector width class)

template<class OriginalType, class ElementType>
__device__ void store_same_length(ElementType* target, Packed128<ElementType> value) {
    int4 bits = value.get_bits();
    switch (sizeof(OriginalType) / sizeof(ElementType)) {
        case 0: *reinterpret_cast<int4*>(target) = bits; // smaller
        case 1: *reinterpret_cast<int4*>(target) = bits; // same size
        case 2: *reinterpret_cast<int2*>(target) = make_int2(bits.x, bits.y); break;
        case 4: *reinterpret_cast<int*>(target) = bits.x; break;
        default: break; //assert(false);
    }
}

// ----------------------------------------------------------------------------
// Helper from ngc92 to create vectors of specific size irrespective of type

template<class ElementType, std::size_t ElementCount>
class alignas(sizeof(ElementType) * ElementCount) GenericVector {
    static_assert(std::is_trivial_v<ElementType>, "Only trivial types are supported");

public:
    GenericVector() = default;
    constexpr __host__ __device__ ElementType& operator[](int index) {
        return values[index];
    }

    constexpr __host__ __device__ const ElementType& operator[](int index) const {
        return values[index];
    }

    static constexpr const std::size_t size = ElementCount;
    static constexpr const std::size_t bytes = ElementCount * sizeof(ElementType);

    static __host__ __device__ GenericVector load(const ElementType* address) {
        return GenericVector(address);
    }

    static __host__ __device__ void store(ElementType* address) {
        // todo
    }

private:
    explicit __host__ __device__ GenericVector(const ElementType* address) {
        if constexpr (bytes % sizeof(int4) == 0) {
            const int4* read_address = reinterpret_cast<const int4*>(address);
            for (int i = 0; i < bytes; i += sizeof(int4)) {
                int4 val = *read_address;
                ++read_address;
                std::memcpy(values + i / sizeof(ElementType), &val, sizeof(int4));
            }
        } else if constexpr (bytes == sizeof(int2)) {
            int2 val = *reinterpret_cast<const int2*>(address);
            std::memcpy(values, &val, sizeof(int2));
        } else if constexpr (bytes == sizeof(int1)) {
            int1 val = *reinterpret_cast<const int1*>(address);
            std::memcpy(values, &val, sizeof(int1));
        } else {
            std::copy(address, address + size, values);
        }
    }

    ElementType values[size];
};

// ----------------------------------------------------------------------------
// CPU code reference

template <bool scaling=SCALING, typename T1, typename T2>
void transpose_cpu(T1* transposed, const T2* input, size_t width, size_t height, T1* copy, float scaling_factor=SCALING_FACTOR) {
    for (size_t y = 0; y < height; y++) {
        for (size_t x = 0; x < width; x++) {
            // note (IN_TYPE) unlike GPU version because T2 is actually always float for simplicity
            float in = (float)((IN_TYPE)input[x + y*width]);
            if constexpr (scaling) { in *= scaling_factor; }

            transposed[y + x * height] = (T1)in;
            copy[x + y*width] = (T1)in;
        }
    }
}

// ----------------------------------------------------------------------------
// GPU kernels for copy

template <bool scaling=SCALING, typename T1, typename T2>
__global__ void copy_naive_kernel(T1 *copy, const T2 *input, size_t N, const float* __restrict__ scale_pointer=d_scaling_factor) {
    size_t n = (blockIdx.x * blockDim.x + threadIdx.x);
    if (n >= N) { return; }
    copy[n] = scaling ? (T1)(*scale_pointer * (float)input[n]) : (T1)input[n];
}

// overly complicated copy & format conversion kernel without store_same_length
// this keeps all loads & stores 128-bit at the cost of more complexity and more register pressure
template <bool scaling=SCALING, bool enable_copy=TRANSPOSE_AND_COPY, typename T1, typename T2>
__global__ void copy128_kernel1(T1 *copy, const T2 *input, size_t N, const float* __restrict__ scale_pointer=d_scaling_factor) {
    size_t n = (blockIdx.x * blockDim.x + threadIdx.x) * Packed128<T1>::size;
    if (n >= N) { return; }

    // note: if sizeof(T1) < sizeof(T2), compiler will skip unused elements of load128
    // so it may turn out to be a load32 or load64
    Packed128<T2> inp128;
    Packed128<T1> out128;
    float scale_factor = scaling ? *scale_pointer : 1.0f;
    #pragma unroll
    for (int o = 0; o < max(1, out128.size/inp128.size); o++) {
        inp128 = load128<T2>(input + n + o*inp128.size);
        #pragma unroll
        for (int k = 0; k < min(inp128.size, out128.size); k++) {
            if constexpr (scaling) {
                out128[k+o*inp128.size] = (T1)((float)inp128[k] * scale_factor);
            }  else {
                out128[k+o*inp128.size] = (T1)inp128[k];
            }
        }
    }
    store128<T1>(copy + n, out128);
}

// simplified copy & format conversion kernel using store_same_length
// keeps the largest format at 128-bit and smallest at 32-bit or 64-bit
template <bool scaling=SCALING, typename T1, typename T2>
__global__ void copy128_kernel2(T1 *copy, const T2 *input, size_t N, const float* __restrict__ scale_pointer=d_scaling_factor) {
    // Calculate the *smallest* of the two vector sizes in terms of elements (both are 128-bit if fully used)
    constexpr size_t vec_size = 16 / ((sizeof(T1) < sizeof(T2)) ? sizeof(T2) : sizeof(T1));

    size_t fewest_elements = min(Packed128<T1>::size, Packed128<T2>::size);
    size_t n = (blockIdx.x * blockDim.x + threadIdx.x) * vec_size;
    if (n >= N) { return; }

    // note: if sizeof(T1) < sizeof(T2), compiler will skip unused elements of load128
    // so it may turn out to be a ldg.32 or ldg.64
    Packed128<T2> inp128;
    Packed128<T1> out128;
    inp128 = load128<T2>(input + n);
    float scale_factor = scaling ? *scale_pointer : 1.0f;
    for (int k = 0; k < vec_size; k++) {
        if constexpr (scaling) {
            out128[k] = (T1)((float)inp128[k] * scale_factor);
        }  else {
            out128[k] = (T1)inp128[k];
        }
    }
    // if sizeof(T2) < sizeof(T1), this will use stg.32 or stg.64 instead of stg.128
    store_same_length<T2,T1>(copy + n, out128);
}

// using ngc92's GenericVector to copy with format conversion
template <bool scaling=SCALING, typename T1, typename T2>
__global__ void copy_vec_kernel3(T1 *copy, const T2 *input, size_t N, const float* __restrict__ scale_pointer=d_scaling_factor) {
    // Calculate the vector size required to use *at least* 128-bit loads and stores (one may be larger)
    constexpr size_t vec_size = 16 / ((sizeof(T1) < sizeof(T2)) ? sizeof(T1) : sizeof(T2));

    size_t n = (blockIdx.x * blockDim.x + threadIdx.x) * vec_size;
    if (n >= N) { return; }

    GenericVector<T2,vec_size> inpV = GenericVector<T2,vec_size>::load(input + n);
    GenericVector<T1,vec_size> outV;
    float scale_factor = scaling ? *scale_pointer : 1.0f;
    for (int k = 0; k < vec_size; k++) {
        outV[k] = scaling ? (T1)((float)inpV[k] * scale_factor) : (T1)inpV[k];
    }
    outV.store(copy + n);
}

// ----------------------------------------------------------------------------
// GPU kernels for transpose

// naive transpose kernel without shared memory or 128-bit load/store
template <bool scaling=SCALING, bool enable_copy=TRANSPOSE_AND_COPY, typename T1, typename T2>
__global__ void transpose_naive_kernel(T1 *transposed, T1* copy, const T2 *input, size_t width, size_t height, const float* __restrict__ scale_pointer=d_scaling_factor) {
    float scale_factor = scaling ? *scale_pointer : 1.0f;
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        T2 in = input[x + y * width];
        T1 out = scaling ? (T1)((float)in * scale_factor) : (T1)in;

        transposed[y + x*height] = out;
        if constexpr (enable_copy) {
            copy[x + y*width] = out;
        }
    }
}

// optimized transpose kernel with shared memory but *without* 128-bit load/store
// originally based on: https://github.com/NVIDIA-developer-blog/code-samples/blob/master/series/cuda-cpp/transpose/transpose.cu
// also see this blog article: https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/
// note that neither of these sources consider less than 32-bit data formats (and associated bank conflicts)
template<size_t BLOCK_ROWS=8UL, size_t TILE_DIM=DEFAULT_TILE, bool scaling=SCALING, bool enable_copy=TRANSPOSE_AND_COPY, typename T1, typename T2>
__global__ void transpose_kernel1(T1 *transposed, T1 *copy, const T2 *input, const float* __restrict__ scale_pointer=d_scaling_factor)
{
    __shared__ T1 tile[TILE_DIM][TILE_DIM+1]; // +1 for bank conflict avoidance
    int width = gridDim.x * TILE_DIM;
    int height = gridDim.y * TILE_DIM;

    float scale_factor = scaling ? *scale_pointer : 1.0f;
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        T2 in = input[x + (y+j)*width];
        T1 out = scaling ? (T1)((float)in * scale_factor) : (T1)in;

        tile[threadIdx.y+j][threadIdx.x] = out;
        if constexpr (enable_copy) {
            copy[x + (y+j)*width] = out; // separate copy with format conversion (on top of the transpose)
        }
    }
    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        // avoiding bank conflicts for 32-bit data types thanks to +1 below (also seems to help sub-32-bit but less)
        transposed[x + (y+j)*height] = tile[threadIdx.x][threadIdx.y + j];
    }
}

// more optimized transpose kernel using 128-bit load/store and shared memory
template<size_t BLOCK_ROWS=8UL, size_t TILE_DIM=DEFAULT_TILE, bool scaling=SCALING, bool enable_copy=TRANSPOSE_AND_COPY, typename T1, typename T2>
__global__ void transpose_kernel2(T1* __restrict__ transposed, T1* __restrict__ copy, const T2* __restrict__ input, const float* __restrict__ scale_pointer=d_scaling_factor)
{
    // no +1 for bank conflict avoidance because:
    // 1) 128-bit shared memory stores need to be aligned to 128-bit boundaries
    // 2) it doesn't help as much with sub-32-bit data types
    __shared__ T1 tile[TILE_DIM][TILE_DIM];
    int width  = gridDim.x * TILE_DIM;
    int height = gridDim.y * TILE_DIM;

    constexpr size_t T1_elements = 16 / sizeof(T1);
    constexpr size_t T2_elements = 16 / sizeof(T2);
    constexpr size_t copy_len = (sizeof(T1) >= sizeof(T2)) ? (sizeof(T1) / sizeof(T2)) : 1;

    float scale_factor = scaling ? *scale_pointer : 1.0f;
    int x = blockIdx.x * TILE_DIM + (threadIdx.x * T2_elements);
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        Packed128<T2> in128 = load128<T2>(input + x + (y+j)*width);
        Packed128<T1> copy128[copy_len];
        for (int k = 0; k < in128.size; k++) {
            T2 in = in128[k];
            T1 out = scaling ? (T1)((float)in * scale_factor) : (T1)in;
            copy128[k/T1_elements][k%T1_elements] = out; // optimised away by compiler if unused
        }

        for (int o = 0; o < copy_len; o++) {
            if constexpr (enable_copy) {
                store_same_length<T2,T1>(copy + x + (y+j)*width + o*T1_elements, copy128[o]);
            }
            size_t tile_offset = (threadIdx.x * T2_elements) + (threadIdx.y+j)*TILE_DIM + o*T1_elements;
            store_same_length<T2,T1>(&tile[0][0] + tile_offset, copy128[o]);
        }
    }
    __syncthreads();

    // we need fewer threads for the write than the read if T1_elements > T2_elements
    if (threadIdx.x >= TILE_DIM / T1_elements) {
        return;
    }

    x = blockIdx.y * TILE_DIM + threadIdx.x * T1_elements;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        // we need more instructions for the write than the read if T2_elements > T1_elements
        for (int o = 0; o < copy_len; o++) {
            Packed128<T1> out128;
            for (int k = 0; k < out128.size; k++) {
                // these are tiny 8-bit loads with loads of bank conflicts for FP8
                // extremely hard to avoid and not a bottleneck when everything else is well optimised
                out128[k] = tile[k + (threadIdx.x + o * blockDim.x) * out128.size][threadIdx.y + j];
            }
            store128<T1>(transposed + x + (o * blockDim.x * out128.size) + (y+j)*height, out128);
        }
    }
}

// ----------------------------------------------------------------------------
// kernel launchers

template <typename T1, typename T2>
void copy_naive(T1 *copy, const T2 *input, size_t width, size_t height, const size_t block_size) {
    size_t N = width * height;
    const dim3 grid_size(ceil_div(N, block_size*block_size));
    const dim3 block_size_(block_size*block_size);
    copy_naive_kernel<<<grid_size, block_size_>>>(copy, input, N);
}

template <typename T1, typename T2>
void copy128_1(T1 *copy, const T2 *input, size_t width, size_t height, const size_t block_size) {
    size_t N = width * height;
    const dim3 grid_size(ceil_div(N, block_size*block_size * 16 / sizeof(T1)));
    const dim3 block_size_(block_size*block_size);
    copy128_kernel1<<<grid_size, block_size_>>>(copy, input, N);
}

template <typename T1, typename T2>
void copy128_2(T1 *copy, const T2 *input, size_t width, size_t height, const size_t block_size) {
    size_t N = width * height;
    size_t fewest_elements = min(Packed128<T1>::size, Packed128<T2>::size);
    const dim3 grid_size(ceil_div(N, block_size*block_size * fewest_elements));
    const dim3 block_size_(block_size*block_size);
    copy128_kernel2<<<grid_size, block_size_>>>(copy, input, N);
}

template <typename T1, typename T2>
void copy_vec_3(T1 *copy, const T2 *input, size_t width, size_t height, const size_t block_size) {
    // Calculate the vector size required to use at least 128-bit for both loads and stores
    constexpr size_t vec_size = 16 / ((sizeof(T1) < sizeof(T2)) ? sizeof(T1) : sizeof(T2));

    size_t N = width * height;
    const dim3 grid_size(ceil_div(N, block_size*block_size * vec_size));
    const dim3 block_size_(block_size*block_size);
    copy_vec_kernel3<<<grid_size, block_size_>>>(copy, input, N);
}

template <typename T1, typename T2>
void transpose_naive(T1 *transposed, const T2 *input, size_t width, size_t height, const size_t block_size, T1 *copy=NULL) {
    const dim3 grid_size(ceil_div(width, block_size), ceil_div(height, block_size));
    const dim3 block_size_(block_size, block_size);
    transpose_naive_kernel<<<grid_size, block_size_>>>(transposed, copy, input, width, height);
}

template <typename T1, typename T2>
void transpose1(T1 *transposed, const T2 *input, size_t width, size_t height, const size_t block_size, T1 *copy=NULL) {
    dim3 grid_size(width / DEFAULT_TILE, height / DEFAULT_TILE);
    dim3 block_size_(DEFAULT_TILE, block_size);

    constexpr bool fused_copy = TRANSPOSE_AND_COPY;

    switch (block_size) {
        case 32: transpose_kernel1<32, DEFAULT_TILE, SCALING, fused_copy><<<grid_size, block_size_>>>(transposed, copy, input); break;
        case 16: transpose_kernel1<16, DEFAULT_TILE, SCALING, fused_copy><<<grid_size, block_size_>>>(transposed, copy, input); break;
        case 8: transpose_kernel1<8, DEFAULT_TILE, SCALING, fused_copy><<<grid_size, block_size_>>>(transposed, copy, input); break;
        case 4: transpose_kernel1<4, DEFAULT_TILE, SCALING, fused_copy><<<grid_size, block_size_>>>(transposed, copy, input); break;
        default: printf("Invalid block size\n"); exit(1);
    }

    if (TRANSPOSE_AND_COPY && !fused_copy) {
        dim3 grid_size(height / DEFAULT_TILE, width / DEFAULT_TILE);
        cudaCheck(cudaGetLastError());
        switch (block_size) {
            case 32: transpose_kernel1<32, DEFAULT_TILE, false, false><<<grid_size, block_size_>>>(copy, copy, transposed); break;
            case 16: transpose_kernel1<16, DEFAULT_TILE, false, false><<<grid_size, block_size_>>>(copy, copy, transposed); break;
            case 8: transpose_kernel1<8, DEFAULT_TILE, false, false><<<grid_size, block_size_>>>(copy, copy, transposed); break;
            case 4: transpose_kernel1<4, DEFAULT_TILE, false, false><<<grid_size, block_size_>>>(copy, copy, transposed); break;
            default: printf("Invalid block size\n"); exit(1);
        }
    }
}

template <typename T1, typename T2>
void transpose2(T1 *transposed, const T2 *input, size_t width, size_t height, const size_t block_size, T1 *copy=NULL) {
    dim3 grid_size(width / DEFAULT_TILE, height / DEFAULT_TILE);
    dim3 block_size_((DEFAULT_TILE * sizeof(T2)) / 16, block_size);

    constexpr bool fused_copy = TRANSPOSE_AND_COPY;

    switch (block_size) {
        case 32: transpose_kernel2<32, DEFAULT_TILE, SCALING, fused_copy><<<grid_size, block_size_>>>(transposed, copy, input); break;
        case 16: transpose_kernel2<16, DEFAULT_TILE, SCALING, fused_copy><<<grid_size, block_size_>>>(transposed, copy, input); break;
        case 8: transpose_kernel2<8, DEFAULT_TILE, SCALING, fused_copy><<<grid_size, block_size_>>>(transposed, copy, input); break;
        case 4: transpose_kernel2<4, DEFAULT_TILE, SCALING, fused_copy><<<grid_size, block_size_>>>(transposed, copy, input); break;
        default: printf("Invalid block size\n"); exit(1);
    }

    if (TRANSPOSE_AND_COPY && !fused_copy) {
        dim3 grid_size(height / DEFAULT_TILE, width / DEFAULT_TILE);
        dim3 block_size_((DEFAULT_TILE * sizeof(T1)) / 16, block_size);
        cudaCheck(cudaGetLastError());
        switch (block_size) {
            case 32: transpose_kernel2<32, DEFAULT_TILE, false, false><<<grid_size, block_size_>>>(copy, copy, transposed); break;
            case 16: transpose_kernel2<16, DEFAULT_TILE, false, false><<<grid_size, block_size_>>>(copy, copy, transposed); break;
            case 8: transpose_kernel2<8, DEFAULT_TILE, false, false><<<grid_size, block_size_>>>(copy, copy, transposed); break;
            case 4: transpose_kernel2<4, DEFAULT_TILE, false, false><<<grid_size, block_size_>>>(copy, copy, transposed); break;
            default: printf("Invalid block size\n"); exit(1);
        }
    }
}

// kernel version dispatch
template <typename T1, typename T2>
void transpose(int kernel_num,
              T1 *transposed, T1 *copy, const T2 *input,
              size_t width, size_t height, size_t block_size) {
    switch (kernel_num) {
        case 0:
            copy_naive(copy, input, width, height, block_size);
            break;
        case 1:
            copy128_1(copy, input, width, height, block_size);
            break;
        case 2:
            copy128_2(copy, input, width, height, block_size);
            break;
        case 3:
            copy_vec_3(copy, input, width, height, block_size);
            break;
        case 10:
            transpose_naive(transposed, input, width, height, block_size, copy);
            break;
        case 11:
            transpose1(transposed, input, width, height, block_size, copy);
            break;
        case 12:
            transpose2(transposed, input, width, height, block_size, copy);
            break;
        default:
            printf("Invalid kernel number\n");
            exit(1);
    }
    cudaCheck(cudaGetLastError());
}

// ----------------------------------------------------------------------------

int main(int argc, const char **argv) {
    setup_main();

    int W = 8192;
    int H = 3072;

    // create host memory of random numbers
    OUT_TYPE* transposed = (OUT_TYPE*)malloc(W * H * sizeof(OUT_TYPE));
    OUT_TYPE* copy = (OUT_TYPE*)malloc(W * H * sizeof(OUT_TYPE));
    OUT_TYPE* out = (OUT_TYPE*)malloc(W * H * sizeof(OUT_TYPE));
    float* input = make_random_float_01(W * H);

    // read kernel_num from command line
    int kernel_num = 12;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);

    // first check the correctness of the kernel
    transpose_cpu(transposed, input, W, H, copy);

    // move to GPU
    IN_TYPE *d_input;
    OUT_TYPE *d_transposed, *d_copy;
    cudaCheck(cudaMalloc(&d_transposed, W * H * sizeof(OUT_TYPE)));
    cudaCheck(cudaMalloc(&d_copy, W * H * sizeof(OUT_TYPE)));
    cudaCheck(cudaMalloc(&d_input, W * H * sizeof(IN_TYPE)));
    cudaCheck(memcpy_convert(d_input, input, W * H));

    float scaling_factor = SCALING_FACTOR;
    cudaCheck(cudaMalloc(&d_scaling_factor, sizeof(float)));
    cudaCheck(cudaMemcpy(d_scaling_factor, &scaling_factor, sizeof(float), cudaMemcpyHostToDevice));

    // time the kernel at different (squares of) block sizes
    int block_sizes[] = {4,8,16,32};
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        printf("Checking block size %d(^2).\n", block_size);
        transpose(kernel_num, d_transposed, d_copy, d_input, W, H, block_size);

        // check copy tensor for copy kernels & for all others in +copy mode
        #if TRANSPOSE_AND_COPY == false
        if (kernel_num < FIRST_TRANSPOSE_KERNEL)
        #endif
            validate_result(d_copy, copy, "copy", W * H, (OUT_TYPE)1e-5f);

        // check transposed tensor for transpose kernels
        if (kernel_num >= FIRST_TRANSPOSE_KERNEL) {
            validate_result(d_transposed, transposed, "transposed", W * H, (OUT_TYPE)1e-5f);
        }
    }
    printf("All results match. Starting benchmarks.\n\n");

    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        int repeat_times = 1000;
        float elapsed_time = benchmark_kernel(repeat_times, transpose<OUT_TYPE, IN_TYPE>,
                                              kernel_num, d_transposed, d_copy, d_input,
                                              W, H, block_size);

        // napkin math: estimate the memory bandwidth achieved
        size_t memory_ops = W * H * (sizeof(IN_TYPE) + sizeof(OUT_TYPE));
        #if TRANSPOSE_AND_COPY == true
        if (kernel_num >= FIRST_TRANSPOSE_KERNEL) {
            memory_ops += W * H * sizeof(OUT_TYPE);
        }
        #endif
        float memory_bandwidth = memory_ops / elapsed_time / 1e6;
        printf("block_size %4d(^2) | time %.4f ms | bandwidth %.2f GB/s\n", block_size, elapsed_time, memory_bandwidth);
    }

    free(out);
    free(copy);
    free(input);
    cudaCheck(cudaFree(d_input));
    cudaCheck(cudaFree(d_copy));
    cudaCheck(cudaFree(d_transposed));
    cudaCheck(cudaFree(d_scaling_factor));
}
