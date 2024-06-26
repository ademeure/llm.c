/*
Matrix Multiplication, with help from cuBLASLt
*/
#include <assert.h>
#include <type_traits>      // std::bool_constant
// llmc internal imports
#include "cuda_common.h"
#include "cuda_utils.cuh"
#include "cublas_common.h"

// GELU is now part of matmul (as it is fused via cuBLASLt for H100)
#include "gelu.cuh"

// ----------------------------------------------------------------------------
// CUDA kernels

template<typename OutFloat, bool UseAuxBuffer>
__global__ void matmul_backward_bias_kernel9(OutFloat* dbias, const floatX* dout, int B, int T, int OC,
                                             std::bool_constant<UseAuxBuffer>) {
    constexpr const int bdx = 4;
    constexpr const int bdy = WARP_SIZE / bdx;
    assert(blockDim.x == bdx);
    assert(blockDim.y == bdy);

    int warp_d = (int)threadIdx.x;
    int warp_c = (int)threadIdx.y;
    int block_d = (int)threadIdx.z;

    const int OC_per_warp = bdy * x128::size;  // 64 at BF16

    int local_oc = warp_c * x128::size;
    int global_oc = blockIdx.x * OC_per_warp + local_oc;

    int local_bt = warp_d + bdx * block_d;
    int bt_per_block = bdx * blockDim.z;

    float accumulators[x128::size];
    for (int k = 0; k < x128::size; k++) {
        accumulators[k] = 0.0f;
    }

    if(global_oc < OC) {
        // sum up over all bt within registers
        for (int idx = blockIdx.y * bt_per_block + local_bt; idx < B * T; idx += gridDim.y * bt_per_block) {
            x128 packed_dout = load128(dout + global_oc + idx*OC);
            for (int k = 0; k < x128::size; k++) {
                accumulators[k] += (float)packed_dout[k];
            }
        }
    }

    __shared__ float sub_results[x128::size][WARP_SIZE][bdy];

    // reduce within-warp results
    for (int k = 0; k < x128::size; k++) {
        float v = accumulators[k];
        v += __shfl_down_sync(0xffffffff, v, 1, 4);
        v += __shfl_down_sync(0xffffffff, v, 2, 4);
        if(warp_d == 0) {
            sub_results[k][block_d][warp_c] = v;
        }
    }
    __syncthreads();

    // block-wide reductions
    for (int k = block_d; k < x128::size; k += blockDim.z) {
        float a = 0.f;
        for (int r = warp_d; r < blockDim.z; r += bdx) {
            float v = sub_results[k][r][warp_c];
            v += __shfl_down_sync(0xffffffff, v, 1, 4);
            v += __shfl_down_sync(0xffffffff, v, 2, 4);
            a += v;
        }
        if(warp_d == 0 && global_oc < OC) {
            if constexpr (!UseAuxBuffer) {
                dbias[global_oc + k] = (OutFloat)(a + (float)dbias[global_oc + k]);
            } else {
                dbias[global_oc + k + blockIdx.y * OC] = a;
            }
        }
    }
}

__global__ void reduce_add_sum_kernel(floatX* dst, const float* src, size_t n, size_t m) {
    const size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * f128::size;
    assert(n % x128::size == 0);
    if (idx < n) {
        f128 acc;
        for(int k = 0; k < f128::size; ++k) {
            acc[k] = 0.f;
        }

        for(int l = 0; l < m; ++l) {
            f128 s = load128(src + idx + n * l);
            for(int k = 0; k < f128::size; ++k) {
                acc[k] += s[k];
            }
        }
        for(int k = 0; k < f128::size; ++k) {
            dst[idx + k] = (floatX) ((float)dst[idx + k] + acc[k]);
        }
    }
}

// ----------------------------------------------------------------------------
// kernel launchers

#define ABSMAX_OUTER_LOOP 4

template <typename T>
__global__ void get_absmax_kernel(const T* inp, float* absmax_scalar, size_t N, float extra_ratio=1.0f) {
    size_t idx = ((blockIdx.x * blockDim.x * ABSMAX_OUTER_LOOP) + threadIdx.x) * x128::size;
    float absmax = 0.0f;

    if (idx < N) {
        for (int i = 0; i < ABSMAX_OUTER_LOOP; i++) {
            for (int k = 0; k < x128::size; ++k) {
                float x = (float)inp[idx + k];
                absmax = max(absmax, fabs(x));
            }
            idx += blockDim.x * x128::size;
        }
    }
    absmax = blockReduce<warpReduceMax>(absmax);

    if (threadIdx.x == 0) {
        absmax = powf(2.0f, floorf(log2f(absmax))); // Round to previous power of 2
        atomicMax((unsigned int*)absmax_scalar, __float_as_uint(absmax));
    }
}

__global__ void get_absmax_kernel(const floatX* inp, float* absmax_scalar, size_t N, float extra_ratio=1.0f) {
    size_t idx = ((blockIdx.x * blockDim.x * ABSMAX_OUTER_LOOP) + threadIdx.x) * x128::size;
    uint absmax_uint = 0;
    x128 packed_inp;

    if (idx < N) {
        #pragma unroll
        for (int i = 0; i < ABSMAX_OUTER_LOOP; i++) {
            x128 packed_inp = load128cs(inp + idx); // load and do not keep in cache
            for(int k = 0; k < packed_inp.size; ++k) {
                uint x = __float_as_uint((float)packed_inp[k] * extra_ratio) & 0x7f800000;
                absmax_uint = max(absmax_uint, x);
            }
            idx += blockDim.x * x128::size;
        }
    }
    // Use inline PTX for redux.sync.max.u32
    uint lane_id = threadIdx.x % 32;
    uint warp_id = threadIdx.x / 32;

    asm volatile("redux.sync.max.u32 %0, %0, 0xff;" : "+r"(absmax_uint));
    __shared__ uint tmp[32];
    if (lane_id == 0) {
        tmp[warp_id] = absmax_uint;
    }
    __syncthreads();
    if (warp_id == 0) {
        absmax_uint = tmp[lane_id];
        // printf max for this blockidx
        if (lane_id == 0) {
            asm volatile("redux.sync.max.u32 %0, %0, 0xff;" : "+r"(absmax_uint));
            atomicMax((unsigned int*)absmax_scalar, absmax_uint);
        }
    }
}




#define FUSED_ABSMAX_OUTER_LOOP 2

__global__ void __launch_bounds__(1024, 2) fused_absmax_scale(__nv_fp8_e4m3* out, const floatX* inp, float* absmax_scalar, unsigned char* absmax_vector, size_t N, float extra_ratio=1.0f) {
    size_t idx = ((blockIdx.x * blockDim.x * FUSED_ABSMAX_OUTER_LOOP) + threadIdx.x) * x128::size;
    size_t elements_per_grid_iteration = (size_t)gridDim.x * (size_t)blockDim.x * x128::size * FUSED_ABSMAX_OUTER_LOOP;

    __shared__ x128 data[1024 * FUSED_ABSMAX_OUTER_LOOP];

    uint absmax_uint = 0;
    __shared__ uint shared_absmax_uint;

    uint lane_id = threadIdx.x % 32;
    uint warp_id = threadIdx.x / 32;

    //printf("idx: %lu, N: %lu, elements_per_grid_iteration: %lu, blockIdx.x: %d, threadIdx.x: %d\n", idx, N, elements_per_grid_iteration, blockIdx.x, threadIdx.x);

    int g = 0;
    //for (; (idx - threadIdx.x * x128::size) < N; g++, idx += elements_per_grid_iteration) {
    if ((idx - (size_t)(threadIdx.x * x128::size)) < N) {
        if (g > 0) {
            // Print EVERYTHING including threadIdx/blockIdx
            //printf("g: %d, idx: %lu, N: %lu, elements_per_grid_iteration: %lu, blockIdx.x: %d, threadIdx.x: %d\n", g, idx, N, elements_per_grid_iteration, blockIdx.x, threadIdx.x);
        }
        if (idx < N) {
            #pragma unroll
            for (int i = 0; i < FUSED_ABSMAX_OUTER_LOOP; i++) {
                x128 packed_inp = load128cs(inp + idx + (i*blockDim.x*x128::size)); // load and do not keep in cache
                data[threadIdx.x + blockDim.x * i] = packed_inp;

                for(int k = 0; k < packed_inp.size; ++k) {
                    uint x = __float_as_uint((float)packed_inp[k] * extra_ratio) & 0x7f800000;
                    absmax_uint = max(absmax_uint, x);
                }
            }
        }
        // Use inline PTX for redux.sync.max.u32
        asm volatile("redux.sync.max.u32 %0, %0, 0xff;" : "+r"(absmax_uint));
        __shared__ uint tmp[32];
        if (lane_id == 0) {
            tmp[warp_id] = absmax_uint;
        }
        asm volatile("bar.cta.sync 0;");
        if (warp_id == 0) {
            absmax_uint = tmp[lane_id];
            // printf max for this blockidx
            asm volatile("redux.sync.max.u32 %0, %0, 0xff;" : "+r"(absmax_uint)); // todo - is this needed or does the compiler always do it?
            absmax_uint = max(atomicMax((unsigned int*)absmax_scalar, absmax_uint), absmax_uint);
            if (lane_id == 0) {
                unsigned char exponent = (absmax_uint >> 23) & 0xff;
                absmax_vector[blockIdx.x + g * gridDim.x] = exponent;
                shared_absmax_uint = absmax_uint;
            }
        }
        // barrier arrive
        asm volatile("bar.cta.sync 0;");
        float scale = 1.0f / __uint_as_float(shared_absmax_uint);

        if (idx >= N) {
            return;
        }

        constexpr int iter_per_fp8 = e4m3_128::size / x128::size;
        constexpr int iter_outer = FUSED_ABSMAX_OUTER_LOOP / iter_per_fp8;
        #pragma unroll iter_outer
        for (int o = 0; o < iter_outer; o++) {
            e4m3_128 packed_out;
            x128 packed_inp[iter_per_fp8];
            #pragma unroll iter_per_fp8
            for (int i = 0; i < iter_per_fp8; i++) {
                packed_inp[i] = data[(threadIdx.x * iter_per_fp8 + i) + (o * blockDim.x * iter_per_fp8)];
            }
            #pragma unroll iter_per_fp8
            for (int i = 0; i < iter_per_fp8; i++) {
                for (int k = 0; k < x128::size; k++) {
                    float x = (float)packed_inp[i][k];
                    packed_out[k + i*x128::size] = (__nv_fp8_e4m3)(x * scale);
                }
            }
            int idx2 = idx + threadIdx.x * (e4m3_128::size - x128::size);
            idx2 += o * blockDim.x * e4m3_128::size;
            store128(out + idx2, packed_out);
        }
    }
}

#define ABSMAX_OUTER_LOOP 4

__global__ void __launch_bounds__(256, 8) fused_absmax_scale_4(__nv_fp8_e4m3* out, const __nv_bfloat16* inp, float* absmax_scalar, unsigned char* absmax_vector, size_t N, float extra_ratio=1.0f) {
    __shared__ uint tmp[8];
    size_t idx = ((blockIdx.x * blockDim.x * (ABSMAX_OUTER_LOOP/2)) + threadIdx.x) * 2 * x128::size;
    uint absmax_uint = 0;
    x128 packed_inp[ABSMAX_OUTER_LOOP];

    if (idx < N) {
        #pragma unroll
        for (int i = 0; i < ABSMAX_OUTER_LOOP; i+=2) {
            if (i == ABSMAX_OUTER_LOOP-2 && idx + i*blockDim.x*x128::size >= N) {
                break;
            }
            packed_inp[i] = load128(inp + idx + i*blockDim.x*x128::size);
            packed_inp[i+1] = load128(inp + idx + i*blockDim.x*x128::size + x128::size); // load and do not keep in cache
            for(int k = 0; k < x128::size; k++) {
                uint x0 = *((unsigned short*)&(packed_inp[i][k])) & 0x7f80;
                uint x1 = *((unsigned short*)&(packed_inp[i+1][k])) & 0x7f80;
                absmax_uint = max(max(absmax_uint, x0), x1);
            }
        }
    }

    asm volatile("redux.sync.max.u32 %0, %0, 0xffffffff;" : "+r"(absmax_uint));
    if ((threadIdx.x % WARP_SIZE) == 0) {
        absmax_uint = __float_as_uint(__uint_as_float(absmax_uint << 16) * extra_ratio);
        tmp[threadIdx.x / WARP_SIZE] = absmax_uint;
    }
    __syncthreads();
    if (threadIdx.x < WARP_SIZE) {
        absmax_uint = tmp[threadIdx.x & 7];
        asm volatile("redux.sync.max.u32 %0, %0, 0xffffffff;" : "+r"(absmax_uint));
        if (threadIdx.x == 0) {
            tmp[0] = absmax_uint;
            tmp[1] = atomicMax((unsigned int*)absmax_scalar, absmax_uint);
        }
    }
    __syncthreads();
    floatX scale = (floatX)(1.0f / __uint_as_float(max(tmp[0], tmp[1])));

    constexpr int iter_per_fp8 = 2;
    constexpr int iter_outer = ABSMAX_OUTER_LOOP / 2;
    if (idx < N) {
        #pragma unroll
        for (int o = 0; o < iter_outer; o++) {
            if (o == iter_outer-1 && idx + o*blockDim.x*e4m3_128::size >= N) {
                return;
            }
            e4m3_128 packed_out;
            #pragma unroll
            for (int i = 0; i < iter_per_fp8; i++) {
                for (int k = 0; k < x128::size; k+=2) {
                    floatX x0 = packed_inp[i+o*2][k] * scale;
                    floatX x1 = packed_inp[i+o*2][k+1] * scale;
                    float2 scaled2 = make_float2((float)x0, (float)x1);
                    __nv_fp8x2_e4m3* packed_fp8x2 = (__nv_fp8x2_e4m3*)&packed_out[k + i*x128::size + 0];
                    *packed_fp8x2 = __nv_fp8x2_e4m3(scaled2);
                }
            }
            store128(out + idx + o*blockDim.x*e4m3_128::size, packed_out);
        }
    }

    if (threadIdx.x == 0) {
        unsigned char exponent = (unsigned char)((*(unsigned short*)&scale) >> 7);
        absmax_vector[blockIdx.x] = exponent;
    }
}

__global__ void __launch_bounds__(256, 8) fused_absmax_scale_3(__nv_fp8_e4m3* out, const __nv_bfloat16* inp, float* absmax_scalar, unsigned char* absmax_vector, size_t N, float extra_ratio=1.0f) {
    __shared__ uint tmp[8];
    size_t idx = ((blockIdx.x * blockDim.x * (ABSMAX_OUTER_LOOP/2)) + threadIdx.x) * 2 * x128::size;
    uint absmax_uint = 0;
    x128 packed_inp[ABSMAX_OUTER_LOOP];

    if (idx + (blockDim.x * ABSMAX_OUTER_LOOP * x128::size) < N) {
        #pragma unroll
        for (int i = 0; i < ABSMAX_OUTER_LOOP; i+=2) {
            packed_inp[i] = load128(inp + idx + i*blockDim.x*x128::size);
            packed_inp[i+1] = load128(inp + idx + i*blockDim.x*x128::size + x128::size); // load and do not keep in cache
            for(int k = 0; k < x128::size; k++) {
                uint x0 = *((unsigned short*)&(packed_inp[i][k])) & 0x7f80;
                uint x1 = *((unsigned short*)&(packed_inp[i+1][k])) & 0x7f80;
                absmax_uint = max(max(absmax_uint, x0), x1);
            }
        }
    } else {
        #pragma unroll
        for (int i = 0; i < ABSMAX_OUTER_LOOP; i+=2) {
            if (i*blockDim.x*x128::size + idx >= N) {
                break;
            }
            packed_inp[i] = load128(inp + idx + i*blockDim.x*x128::size);
            packed_inp[i+1] = load128(inp + idx + i*blockDim.x*x128::size + x128::size); // load and do not keep in cache
            for(int k = 0; k < x128::size; k++) {
                uint x0 = *((unsigned short*)&(packed_inp[i][k])) & 0x7f80;
                uint x1 = *((unsigned short*)&(packed_inp[i+1][k])) & 0x7f80;
                absmax_uint = max(max(absmax_uint, x0), x1);
            }
        }
    }

    asm volatile("redux.sync.max.u32 %0, %0, 0xffffffff;" : "+r"(absmax_uint));
    if ((threadIdx.x % WARP_SIZE) == 0) {
        absmax_uint = __float_as_uint(__uint_as_float(absmax_uint << 16) * extra_ratio);
        tmp[threadIdx.x / WARP_SIZE] = absmax_uint;
    }
    __syncthreads();
    if (idx >= N) {
        return;
    }
    if (threadIdx.x < WARP_SIZE) {
        absmax_uint = tmp[threadIdx.x & 7];
        asm volatile("redux.sync.max.u32 %0, %0, 0xffffffff;" : "+r"(absmax_uint));
        if (threadIdx.x == 0) {
            absmax_uint = max(absmax_uint, atomicMax((unsigned int*)absmax_scalar, absmax_uint));
            tmp[0] = absmax_uint;
        }
    }
    __syncthreads();
    floatX scale = (floatX)(1.0f / __uint_as_float(tmp[0]));

    constexpr int iter_per_fp8 = 2;
    constexpr int iter_outer = ABSMAX_OUTER_LOOP / 2;
    #pragma unroll
    for (int o = 0; o < iter_outer; o++) {
        e4m3_128 packed_out;
        #pragma unroll
        for (int i = 0; i < iter_per_fp8; i++) {
            for (int k = 0; k < x128::size; k+=2) {
                floatX x0 = packed_inp[i+o*2][k] * scale;
                floatX x1 = packed_inp[i+o*2][k+1] * scale;
                float2 scaled2 = make_float2((float)x0, (float)x1);
                __nv_fp8x2_e4m3* packed_fp8x2 = (__nv_fp8x2_e4m3*)&packed_out[k + i*x128::size + 0];
                *packed_fp8x2 = __nv_fp8x2_e4m3(scaled2);
            }
        }
        if (idx + o*blockDim.x*e4m3_128::size >= N) {
            break;
        }
        store128(out + idx + o*blockDim.x*e4m3_128::size, packed_out);
    }

    if (threadIdx.x == 0) {
        unsigned char exponent = (unsigned char)((*(unsigned short*)&scale) >> 7);
        absmax_vector[blockIdx.x] = exponent;
    }
}



// block size must be equal to fused_absmax_scale block size, divided by 2 (FP8 per BF16), multiplied by ABSMAX_OUTER_LOOP
// ==> (256 / 2) * 4 = 512
__global__ void __launch_bounds__(256, 8) rescale_kernel_two(__nv_fp8_e4m3* in_out, float* absmax_scalar, unsigned char* absmax_vector, size_t N) {
    float scale = 1.0f / __uint_as_float(*((unsigned int*)absmax_scalar));
    unsigned char target_exponent = (unsigned char)((*(unsigned int*)&scale) >> 23);
    unsigned char current_exponent = absmax_vector[blockIdx.x];
    if (target_exponent == current_exponent) {
        return;
    }
    size_t idx = ((blockIdx.x * blockDim.x) + threadIdx.x) * e4m3_128::size * 2;
    int exponent_difference = target_exponent - current_exponent;
    float scale_factor = exp2f((float)exponent_difference);

    e4m3_128 packed_data[2];
    for (int i = 0; i < 2; i++) {
        packed_data[i] = load128(in_out + idx + i*e4m3_128::size);
        for (int k = 0; k < e4m3_128::size; k++) {
            float x = (float)packed_data[i][k];
            x *= scale_factor;
            packed_data[i][k] = (__nv_fp8_e4m3)x;
        }
    }
    store128(in_out + idx, packed_data[0]);
    store128(in_out + idx + e4m3_128::size, packed_data[1]);
}


// block size must be equal to fused_absmax_scale block size, divided by 2 (FP8 per BF16), multiplied by ABSMAX_OUTER_LOOP
// ==> (256 / 2) * 4 = 512
__global__ void __launch_bounds__(512, 4) rescale_kernel(__nv_fp8_e4m3* in_out, float* absmax_scalar, unsigned char* absmax_vector, size_t N) {
    float scale = 1.0f / __uint_as_float(*((unsigned int*)absmax_scalar));
    unsigned char target_exponent = (unsigned char)((*(unsigned int*)&scale) >> 23);
    unsigned char current_exponent = absmax_vector[blockIdx.x];
    if (target_exponent == current_exponent) {
        return;
    }
    size_t idx = ((blockIdx.x * blockDim.x) + threadIdx.x) * e4m3_128::size;
    int exponent_difference = target_exponent - current_exponent;
    float scale_factor = exp2f((float)exponent_difference);

    e4m3_128 packed_data = load128(in_out + idx);
    for (int k = 0; k < e4m3_128::size; k++) {
        float x = (float)packed_data[k];
        x *= scale_factor;
        packed_data[k] = (__nv_fp8_e4m3)x;
    }
    store128(in_out + idx, packed_data);
}

// block size must be equal to fused_absmax_scale block size, divided by 2 (FP8 per BF16), multiplied by ABSMAX_OUTER_LOOP
// ==> (256 / 2) * 4 = 512
template <int consecutive=8>
__global__ void __launch_bounds__(512, 4) rescale_kernel_multi(__nv_fp8_e4m3* in_out, float* absmax_scalar, unsigned char* absmax_vector, size_t N) {
    float scale = 1.0f / __uint_as_float(*((unsigned int*)absmax_scalar));
    unsigned char target_exponent = (unsigned char)((*(unsigned int*)&scale) >> 23);
    unsigned char current_exponent_multi = absmax_vector[(blockIdx.x * consecutive) + (threadIdx.x % consecutive)];
    bool same_exponent = (target_exponent == current_exponent_multi);

    // Check if same_exponent is true for all threads in warp
    bool all_same_exponent = __all_sync(0xffffffff, same_exponent);
    if (all_same_exponent) {
        return;
    }

    for (int i = 0; i < consecutive; i++) {
        // Get current_exponent from thread i of warp using __shfl_sync
        unsigned char current_exponent = __shfl_sync(0xffffffff, current_exponent_multi, i, consecutive);
        if (target_exponent == current_exponent) {
            continue;
        }

        size_t idx = (((blockIdx.x * consecutive + i) * blockDim.x) + threadIdx.x) * e4m3_128::size;

        int exponent_difference = target_exponent - current_exponent;
        float scale_factor = exp2f((float)exponent_difference);

        if (threadIdx.x == 0) {
            //printf("idx: %lu, N: %lu, block_idx: %d, target_exponent: %d, current_exponent: %d, exponent_difference: %d, scale_factor: %f\n", idx, N, blockIdx.x, target_exponent, current_exponent, exponent_difference, scale_factor);
        }

        e4m3_128 packed_data = load128(in_out + idx);
        for (int k = 0; k < e4m3_128::size; k++) {
            float x = (float)packed_data[k];
            x *= scale_factor;
            packed_data[k] = (__nv_fp8_e4m3)x;
        }
        store128(in_out + idx, packed_data);
    }
}




template <typename T>
void get_absmax(const T* inp, float* absmax_scalar, bool memset, size_t N, float extra_ratio=1.0f, cudaStream_t stream=0) {
    NVTX_RANGE_FN();
    size_t block_size = 1024;
    assert((N % (x128::size * ABSMAX_OUTER_LOOP)) == 0);
    size_t grid_size = CEIL_DIV(N, block_size * x128::size * ABSMAX_OUTER_LOOP);

    if (memset) {
        cudaMemset(absmax_scalar, 0, sizeof(float));
    }
    //get_absmax_kernel<<<grid_size, block_size, 0, stream>>>(inp, absmax_scalar, N, extra_ratio);

    //grid_size = 2 * deviceProp.multiProcessorCount;
    get_absmax_kernel<<<grid_size, block_size, 0, stream>>>(inp, absmax_scalar, N, extra_ratio);
    cudaCheck(cudaGetLastError());
}


void get_absmax_and_scale(__nv_fp8_e4m3* out, const floatX* inp, float* absmax_global, unsigned char* absmax_vector, bool memset, size_t N, float extra_ratio=1.0f, cudaStream_t stream=0) {
    NVTX_RANGE_FN();
    size_t block_size = 256;
    assert((N % (x128::size * ABSMAX_OUTER_LOOP)) == 0 || (N % (x128::size * (ABSMAX_OUTER_LOOP-2))) == 0);
    size_t grid_size = CEIL_DIV(N, block_size * x128::size * ABSMAX_OUTER_LOOP);

    if (memset) {
        cudaMemset(absmax_global, 0, sizeof(float));
    }

    /*
    size_t bytes = 256 + grid_size;
    cudaStreamAttrValue stream_attribute;                                         // Stream level attributes data structure
    stream_attribute.accessPolicyWindow.base_ptr  = reinterpret_cast<void*>(absmax_global); // Global Memory data pointer
    stream_attribute.accessPolicyWindow.num_bytes = bytes;                    // Number of bytes for persistence access.
    stream_attribute.accessPolicyWindow.hitRatio  = 1.0;                          // Hint for cache hit ratio
    stream_attribute.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting; // Type of access property on cache hit
    stream_attribute.accessPolicyWindow.missProp  = cudaAccessPropertyNormal;  // Type of access property on cache miss.
    cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);
    */

    fused_absmax_scale_3<<<grid_size, block_size, 0, stream>>>(out, inp, absmax_global, absmax_vector, N, extra_ratio);
    cudaCheck(cudaGetLastError());

    /*
    stream_attribute.accessPolicyWindow.base_ptr  = reinterpret_cast<void*>(absmax_global); // Global Memory data pointer
    stream_attribute.accessPolicyWindow.num_bytes = bytes;                    // Number of bytes for persistence access.
    stream_attribute.accessPolicyWindow.hitRatio  = 1.0;                          // Hint for cache hit ratio
    stream_attribute.accessPolicyWindow.hitProp   = cudaAccessPropertyNormal; // Type of access property on cache hit
    stream_attribute.accessPolicyWindow.missProp  = cudaAccessPropertyNormal;  // Type of access property on cache miss.
    cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);
    */

    if (grid_size > deviceProp.multiProcessorCount * 20 && (grid_size % 8) == 0) {
        rescale_kernel_multi<8><<<grid_size/8, 512, 0, stream>>>(out, absmax_global, absmax_vector, N);
    } else {
        rescale_kernel<<<grid_size, 512, 0, stream>>>(out, absmax_global, absmax_vector, N);
    }
    cudaCheck(cudaGetLastError());
}

#define SCALE_OUTER_LOOP 2

__global__ void scale_tensor_kernel(__nv_fp8_e4m3* out, const floatX* inp, float* scaleptr, bool reciprocal) {
    size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size * SCALE_OUTER_LOOP;

    float scale = *scaleptr;
    if(reciprocal) {
        scale = 1.0f / scale;
    }

    constexpr int iter_per_fp8 = e4m3_128::size / x128::size;
    constexpr int iter_outer = SCALE_OUTER_LOOP / iter_per_fp8;
    #pragma unroll iter_outer
    for (int o = 0; o < iter_outer; o++) {
        e4m3_128 packed_out;
        size_t start_idx = idx;
        #pragma unroll iter_per_fp8
        for (int i = 0; i < iter_per_fp8; i++) {
            x128 packed_inp = load128cs(inp + idx); // load and do not keep in cache
            #pragma unroll x128::size
            for(int k = 0; k < x128::size; ++k) {
                float x = (float)packed_inp[k];
                packed_out[k + i*x128::size] = (__nv_fp8_e4m3)(x * scale);
            }
            idx += x128::size;
        }
        store128(out + start_idx, packed_out);
    }
}

__global__ void scale_tensor_kernel(floatX* out, const floatX* inp, float* scaleptr, bool reciprocal) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size * SCALE_OUTER_LOOP;

    float scale = *scaleptr;
    if(reciprocal) {
        scale = 1.0f / scale;
    }

    for (int i = 0; i < SCALE_OUTER_LOOP; i++) {
        x128 packed_out;
        x128 packed_inp = load128cs(inp + idx); // load and do not keep in cache
        for(int k = 0; k < packed_inp.size; ++k) {
            float x = (float)packed_inp[k];
            packed_out[k] = (floatX)(x * scale);
        }
        store128(out + idx, packed_out);
    }
}

template <typename T>
__global__ void scale_tensor_kernel(T* out, const floatX* inp, float* scaleptr, bool reciprocal) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size * SCALE_OUTER_LOOP;

    float scale = *scaleptr;
    if(reciprocal) {
        scale = 1.0f / scale;
    }

    for (int i = 0; i < SCALE_OUTER_LOOP; i++, idx += x128::size) {
        x128 packed_inp = load128cs(inp + idx); // load and do not keep in cache
        for(int k = 0; k < packed_inp.size; ++k) {
            float x = (float)packed_inp[k];
            out[idx + k] = (T)(x * scale);
        }
    }
}

template <typename T>
void scale_tensor(T* out, const floatX* inp, float* scaleptr, bool reciprocal, int N, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 64;
    assert(N % (block_size * x128::size * SCALE_OUTER_LOOP) == 0);
    const int grid_size = CEIL_DIV(N, block_size * x128::size * SCALE_OUTER_LOOP);

    generate_analysis(inp, N, "pre_scale_tensor");
    scale_tensor_kernel<<<grid_size, block_size, 0, stream>>>(out, inp, scaleptr, reciprocal);
    cudaCheck(cudaGetLastError());
    generate_analysis(out, N, "post_scale_tensor");
}

//#define FORCE_FP8_MATMUL
int this_time = 0;

// https://docs.nvidia.com/cuda/cublas/#cublasltmatmul
void matmul_cublaslt(floatX* d, const floatX* a, const floatX* b, const floatX* bias,
                     int m, int n, int k, cudaStream_t stream, bool transA=true, bool transB=false,
                     int batch_count=0, size_t strideA=0, size_t strideB=0, size_t strideOut=0,
                     bool accumulate=false, floatX* pre_gelu=NULL, bool backward=false, bool allow_fp8=true) {
    NVTX_RANGE_FN();
    bool has_bias = (bias != NULL);
    bool has_gelu = (pre_gelu != NULL);

    // check alignments for all pointers (only need 16 bytes alignment in some modes but it never hurts to be aligned perf-wise!)
    if(((uintptr_t)a % 16) != 0 || ((uintptr_t)b % 16) != 0 || ((uintptr_t)d % 16) != 0 || ((uintptr_t)bias % 16) != 0) {
        printf("One of the pointers is not aligned (cuBLASLt requirement)!\n");
        exit(EXIT_FAILURE);
    }

    cublasLtMatmulDesc_t operationDesc;
    cublasCheck(cublasLtMatmulDescCreate(&operationDesc, cublas_compute, CUDA_R_32F)); // FP16 if CUBLAS_COMPUTE_16F

    int returnedResults = 0;
    cublasLtMatrixLayout_t ALayout;
    cublasLtMatrixLayout_t BLayout;
    cublasLtMatrixLayout_t DLayout;
    cublasLtMatrixLayout_t CLayout;
    cublasLtMatmulPreference_t preference;
    cublasLtMatmulHeuristicResult_t heuristic;
    cublasOperation_t opNoTranspose = CUBLAS_OP_N;
    cublasOperation_t opTranspose = CUBLAS_OP_T;

    auto input_precision = CUBLAS_LOWP;
    #ifdef FORCE_FP8_MATMUL
    if (transA && !transB && batch_count == 0 && allow_fp8) {
        input_precision = CUDA_R_8F_E4M3;
        int8_t fast_accum = 1;
        cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fast_accum, sizeof(fast_accum)));

        if (has_gelu) {
            //cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_AMAX_POINTER, (void*)huge_scratch, sizeof(huge_scratch)));
        } else {
            //cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_AMAX_D_POINTER, (void*)&huge_scratch, sizeof(huge_scratch)));
        }

        // Get absmax for a and b
        size_t absmax_size = CEIL_DIV(8*max(n*k,m*k), 256 * 8 * x128::size);

        size_t rounded_to_128a = CEIL_DIV(128+absmax_size, 128) * 128;
        size_t rounded_to_128b = CEIL_DIV(128+absmax_size+m*k, 128) * 128;

        unsigned char* absmax_vector = &((unsigned char*)huge_scratch)[128];
        float* absmax_a = &((float*)huge_scratch)[0];
        float* absmax_b = &((float*)huge_scratch)[1];
        __nv_fp8_e4m3* a_fp8 = &((__nv_fp8_e4m3*)huge_scratch)[rounded_to_128a];
        __nv_fp8_e4m3* b_fp8 = &((__nv_fp8_e4m3*)huge_scratch)[rounded_to_128b];

        this_time++;
        /*if (this_time == 6000) {
            printf("OVER 6000!!!\n\n\n");
        }*/
        get_absmax_and_scale(a_fp8, a, absmax_a, absmax_vector, true, m*k, 1.0f/128.0f, stream);
        get_absmax_and_scale(b_fp8, b, absmax_b, absmax_vector, true, n*k, 1.0f/128.0f, stream);

        /*
        floatX* b_host = (floatX*)malloc(k*n*sizeof(floatX));
        cudaMemcpy(b_host, b, k*n*sizeof(floatX), cudaMemcpyDeviceToHost);

        get_absmax(b, absmax_b, true, k*n, 1.0/128.0f, stream);
        scale_tensor<__nv_fp8_e4m3>(b_fp8, b, absmax_b, true, k*n, stream);

        //get_absmax(b, absmax_a, true, m*k, 1.0/128.0f, stream);
        //scale_tensor<__nv_fp8_e4m3>(a_fp8, b, absmax_a, true, m*k, stream);

        get_absmax_and_scale(a_fp8, b, absmax_a, absmax_vector, true, k*n, 1.0f/128.0f, stream);

        cudaStreamSynchronize(stream);
        float absmax_a_host, absmax_b_host;
        cudaMemcpy(&absmax_a_host, absmax_a, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&absmax_b_host, absmax_b, sizeof(float), cudaMemcpyDeviceToHost);
        printf("absmax_a: %f, absmax_b: %f\n", absmax_a_host, absmax_b_host);

        // Compare all the differences between b_fp8 and a_fp8
        cudaDeviceSynchronize();
        __nv_fp8_e4m3* a_fp8_host = (__nv_fp8_e4m3*)malloc(k*n*sizeof(__nv_fp8_e4m3));
        __nv_fp8_e4m3* b_fp8_host = (__nv_fp8_e4m3*)malloc(k*n*sizeof(__nv_fp8_e4m3));
        cudaMemcpy(a_fp8_host, a_fp8, k*n*sizeof(__nv_fp8_e4m3), cudaMemcpyDeviceToHost);
        cudaMemcpy(b_fp8_host, b_fp8, k*n*sizeof(__nv_fp8_e4m3), cudaMemcpyDeviceToHost);
        // Get B in BF16 from GPU to CPU

        for (int i = 0; i < k*n; i += 1024) {
            if ((float)a_fp8_host[i] != (float)b_fp8_host[i]) {
                printf("\na_fp8_host[%d]: %f, b_fp8_host[%d]: %f ==> %f (BF16: %f)\n", i, (float)a_fp8_host[i], i, (float)b_fp8_host[i], (float)a_fp8_host[i] / (float)b_fp8_host[i], (float)b_host[i]);
            } else {
                printf("... Same[%d] ...", i);
            }
        }

        // get absmax of a_fp8
        float* absmax_a_fp8 = &((float*)&huge_scratch)[2];
        get_absmax(a_fp8, absmax_a_fp8, true, m*k, 1.0f, stream);
        cudaStreamSynchronize(stream);
        float absmax_a_fp8_host;
        cudaMemcpy(&absmax_a_fp8_host, absmax_a_fp8, sizeof(float), cudaMemcpyDeviceToHost);
        printf("absmax_a_fp8: %f\n", absmax_a_fp8_host);
        */


        a = (floatX*)a_fp8;
        b = (floatX*)b_fp8;

        // Set CUBLASLT_MATMUL_DESC_A_SCALE_POINTER
        cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &absmax_a, sizeof(absmax_a)));
        cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &absmax_b, sizeof(absmax_b)));

        strideA /= (sizeof(floatX) / sizeof(__nv_fp8_e4m3));
        strideB /= (sizeof(floatX) / sizeof(__nv_fp8_e4m3));
    }
    #endif

    cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, (transA)  ? &opTranspose : &opNoTranspose,   sizeof(opTranspose)));
    cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, (transB) ? &opTranspose   : &opNoTranspose, sizeof(opNoTranspose)));

    // Define matrix layouts
    if (transA) {
        cublasCheck(cublasLtMatrixLayoutCreate(&ALayout, input_precision, k, m, k));
    } else {
        cublasCheck(cublasLtMatrixLayoutCreate(&ALayout, input_precision, m, k, m));
    }
    if (transB) {
        cublasCheck(cublasLtMatrixLayoutCreate(&BLayout, input_precision, n, k, n));
    } else {
        cublasCheck(cublasLtMatrixLayoutCreate(&BLayout, input_precision, k, n, k));
    }
    // cuBLASLt requires C in FP8 mode to be BF16 or FP32...
    cublasCheck(cublasLtMatrixLayoutCreate(&CLayout, (sizeof(floatX) == 1) ? CUDA_R_16BF : CUBLAS_LOWP, m, n, m));
    cublasCheck(cublasLtMatrixLayoutCreate(&DLayout, CUBLAS_LOWP, m, n, m));

    if (batch_count) {
        cublasCheck(cublasLtMatrixLayoutSetAttribute(ALayout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
        cublasCheck(cublasLtMatrixLayoutSetAttribute(BLayout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
        cublasCheck(cublasLtMatrixLayoutSetAttribute(CLayout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
        cublasCheck(cublasLtMatrixLayoutSetAttribute(DLayout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));

        cublasCheck(cublasLtMatrixLayoutSetAttribute(ALayout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideA, sizeof(strideA)));
        cublasCheck(cublasLtMatrixLayoutSetAttribute(BLayout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideB, sizeof(strideB)));
        cublasCheck(cublasLtMatrixLayoutSetAttribute(CLayout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideOut, sizeof(strideOut)));
        cublasCheck(cublasLtMatrixLayoutSetAttribute(DLayout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideOut, sizeof(strideOut)));
    }

    // create a preference handle with specified max workspace
    cublasCheck(cublasLtMatmulPreferenceCreate(&preference));
    cublasCheck(cublasLtMatmulPreferenceSetAttribute(preference,
    CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &cublaslt_workspace_size, sizeof(cublaslt_workspace_size)));

    cublasLtEpilogue_t epilogue;
    if (has_gelu) {
        int64_t gelu_ld = m; // todo - is this affected by anything else?
        cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, &gelu_ld, sizeof(gelu_ld)));
        cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, &pre_gelu, sizeof(pre_gelu)));
        if (backward) {
            assert(!has_bias); // not the same backward matmul that does GELU and bias
            epilogue = CUBLASLT_EPILOGUE_DGELU;
        } else {
            epilogue = has_bias ? CUBLASLT_EPILOGUE_GELU_AUX_BIAS : CUBLASLT_EPILOGUE_GELU_AUX;
        }
    } else if(has_bias){
        epilogue = backward ? CUBLASLT_EPILOGUE_BGRADB : CUBLASLT_EPILOGUE_BIAS;
    } else {
        epilogue = CUBLASLT_EPILOGUE_DEFAULT;
    }
    cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));

    if (has_bias) {
        // cuBLASLt requires bias in FP8 mode to be BF16...
        cublasDataType_t bias_data_type = (sizeof(floatX) == 1) ? CUDA_R_16BF : CUBLAS_LOWP; // force BF16 bias for FP8 mode
        cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_data_type, sizeof(bias_data_type)));
        cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
    }

    // Set scale type to FP32 (needs to be FP16 if and only if using CUBLAS_COMPUTE_16F)
    const float alpha = 1.0f, beta = accumulate ? 1.0f : 0.0f;
    cublasDataType_t scale_type = CUDA_R_32F;
    cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_SCALE_TYPE, &scale_type, sizeof(scale_type)));

    // find a suitable algorithm
    cublasLtMatmulAlgoGetHeuristic(cublaslt_handle, operationDesc,
        ALayout, BLayout, CLayout, DLayout,
        preference, 1, &heuristic, &returnedResults);

    cudaCheck(cudaGetLastError());

    if (returnedResults == 0) {
        printf("No cuBLASLt algorithm: n: %d, k: %d, m: %d, bias: %d\n", n, k, m, has_bias);
        exit(EXIT_FAILURE);
    }

    // call the matmul
    cublasCheck(cublasLtMatmul(cublaslt_handle, operationDesc,
        &alpha, a, ALayout, b, BLayout, &beta,
        d, CLayout, d, DLayout, &heuristic.algo,
        cublaslt_workspace, cublaslt_workspace_size, stream));

    // cleanups
    cublasCheck(cublasLtMatmulPreferenceDestroy(preference));
    cublasCheck(cublasLtMatmulDescDestroy(operationDesc));
    cublasCheck(cublasLtMatrixLayoutDestroy(ALayout));
    cublasCheck(cublasLtMatrixLayoutDestroy(BLayout));
    cublasCheck(cublasLtMatrixLayoutDestroy(CLayout));
    cublasCheck(cublasLtMatrixLayoutDestroy(DLayout));

    //printf("Yay\n");

    // print m, n, k
    //printf("m: %d, n: %d, k: %d\n", m, n, k);
}

#define TILE_DIM 32
#define BLOCK_ROWS 4

template<typename T>
__global__ void transposeNoBankConflicts(T *odata, const T *idata)
{
  __shared__ T tile[TILE_DIM][TILE_DIM+1];

  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int width = gridDim.x * TILE_DIM;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
     tile[threadIdx.y+j][threadIdx.x] = idata[(y+j)*width + x];

  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
  y = blockIdx.x * TILE_DIM + threadIdx.y;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
     odata[(y+j)*width + x] = tile[threadIdx.x][threadIdx.y + j];
}

// this is a slightly confusing wrapper that we should maybe get rid of...
 // m=OC, n=B*T, k=C with a=weight and b=inp
void matmul_forward_cublaslt(floatX* out,
                     floatX* inp, floatX* weight, floatX* bias,
                     int B, int T, int C, int OC, cudaStream_t stream,
                     floatX* pre_gelu=NULL) {

    generate_analysis(weight, OC*C, "matmul_fwd_weight");
    generate_analysis(inp, B*T*C, "matmul_fwd_inp");
    if (bias) {
        generate_analysis(bias, OC, "matmul_fwd_bias");
    }

    // Only fuse GELU for H100+ as cuBLAS seems to be inefficient for fused GELU on Ada/Ampere (???)
    if (deviceProp.major < 9 && pre_gelu) {
        matmul_cublaslt(pre_gelu, weight, inp, bias, OC, B*T, C, stream, true, false, 0, 0, 0, 0, false, NULL, false, true);
        gelu_forward(out, pre_gelu, B*T*OC, stream);
    } else {
        matmul_cublaslt(out, weight, inp, bias, OC, B*T, C, stream, true, false, 0, 0, 0, 0, false, pre_gelu, false, true);
        if (pre_gelu) {
            generate_analysis(pre_gelu, OC*B*T, "matmul_fwd_act_pre_gelu");
        }
    }
    generate_analysis(out, OC*B*T, "matmul_fwd_act_out");
}

void matmul_backward(floatX* dinp, floatX* dweight, floatX* dbias,
                     floatX* dout, floatX* inp, floatX* weight,
                     float* dbias_buffer,
                     int B, int T, int C, int OC, cudaStream_t stream,
                    floatX* pre_gelu=NULL) {
    NVTX_RANGE_FN();

    generate_analysis(dout, (size_t)B*(size_t)T*(size_t)OC, "matmul_bwd_in_dout");
    generate_analysis(weight, C*OC, "matmul_bwd1_in_w");

    // backward to bias, if given, does a +=
    // TODO: not fusing in cuBLASLt as it reduces accuracy in BF16 and it's barely faster...
    // (but it is supported if this code is disabled as dbias is passed to matmul_cublaslt below)
    if (dbias != NULL) {
        // Each warp is responsible for 8 * "x128::size" = 64 OCs at BF16 (OC must be a multiple of 64!)
        // Block size is 1024 | 768 threads (32|24 warps) and we reduce those values into 1 at the end

        const int block_size = deviceProp.maxThreadsPerMultiProcessor == 1536 ? 768 : 1024;

        dim3 block_dim = {4, 8, (unsigned)block_size/WARP_SIZE};
        const int OC_per_warp = block_dim.y * x128::size; // 64 at BF16
        const int grid_size_x = CEIL_DIV(OC, OC_per_warp); // e.g. 12 horizontal blocks for 768 OCs at BF16
        const int grid_size_y = max(1, deviceProp.maxThreadsPerMultiProcessor * deviceProp.multiProcessorCount / (block_size * grid_size_x)); // full GPU!

        // If we have enough OC that we don't need cross-block reductions, we can skip the bias_buffer accumulation
        // and write results directly to the output.
        if(grid_size_y == 1) {
            matmul_backward_bias_kernel9<<<dim3(grid_size_x, grid_size_y), block_dim, 0, stream>>>(dbias, dout, B, T, OC, std::bool_constant<false>{});
            cudaCheck(cudaGetLastError());
        } else {
            // kernel 9 overwrites temp buffer, so no need to memset
            matmul_backward_bias_kernel9<<<dim3(grid_size_x, grid_size_y), block_dim, 0, stream>>>(dbias_buffer, dout, B, T, OC, std::bool_constant<true>{});
            cudaCheck(cudaGetLastError());
            reduce_add_sum_kernel<<<CEIL_DIV(OC, 256 * f128::size), 256, 0, stream>>>(dbias, dbias_buffer, OC, grid_size_y);
            cudaCheck(cudaGetLastError());
        }
        generate_analysis(dbias, OC, "matmul_bwd_wgrad_bias");
        dbias = NULL; // prevent dbias calculation from also being fused in matmul_cublaslt below
    }

    // Let's try to transpose weight
    floatX* weight_transposed = (floatX*)huge_scratch;

    printf("C: %d, OC: %d\n", C, OC);
    dim3 dimGrid(OC/(TILE_DIM), C/(TILE_DIM), 1);
    dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);
    transposeNoBankConflicts<<<dimGrid, dimBlock>>>((float*)weight_transposed, (float*)weight);
    cudaCheck(cudaGetLastError());

    // Get both weight and weight_transposed to CPU
    floatX* weight_host = (floatX*)malloc(C*OC*sizeof(floatX));
    cudaMemcpy(weight_host, weight, C*OC*sizeof(floatX), cudaMemcpyDeviceToHost);
    floatX* weight_transposed_host = (floatX*)malloc(C*OC*sizeof(floatX));
    cudaMemcpy(weight_transposed_host, weight_transposed, C*OC*sizeof(floatX), cudaMemcpyDeviceToHost);
    // Printf a 4x4 square
    if (weight_host[0] != (floatX)0.0f) {
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                printf("%d/%d => weight[%d]: %f, weight_transposed[%d]: %f\n", i, j, i*OC+j, (float)weight_host[i*OC+j], i*OC+j, (float)weight_transposed_host[i*OC+j]);
            }
        }
    }

    // backward to input, uses = in the backward pass (set the gradient)
    //matmul_cublaslt(dinp, weight, dout, NULL, C, B*T, OC, stream, false, false, 0, 0, 0, 0, false, /*pre_gelu*/ NULL, true, false);
    matmul_cublaslt(dinp, weight_transposed, dout, NULL, C, B*T, OC, stream, true, false, 0, 0, 0, 0, false, /*pre_gelu*/ NULL, true, false);

    generate_analysis(dinp, (size_t)C*(size_t)B*(size_t)T, "matmul_bwd1_agrad_dinp");
    generate_analysis(inp, (size_t)C*(size_t)B*(size_t)T, "matmul_bwd2_in_inp");

    if (pre_gelu) {
        // TODO: not fusing in cuBLASLt as it reduces accuracy (/performance?) with BF16 :(
        gelu_backward_inplace(dinp, pre_gelu, B*T*C, stream);
    }
/*
    floatX* inp_transposed = &((floatX*)huge_scratch)[512*1024*1024];
    dim3 dimGrid2(C/TILE_DIM, B*T/TILE_DIM, 1);
    dim3 dimBlock2(TILE_DIM, BLOCK_ROWS, 1);
    //transposeNoBankConflicts<<<dimGrid2, dimBlock2>>>((floatX*)inp_transposed, (floatX*)inp);

    floatX* dout_transposed = &((floatX*)huge_scratch)[768*1024*1024];
    dim3 dimGrid3(B*T/TILE_DIM, OC/TILE_DIM, 1);
    dim3 dimBlock3(TILE_DIM, BLOCK_ROWS, 1);
    //transposeNoBankConflicts<<<dimGrid2, dimBlock2>>>((floatX*)dout_transposed, (floatX*)dout);
*/

    // backward to weight, uses += in the backward pass (accumulate the gradient) by setting alpha=one
    // but if this is the 1st micro-step, then dweight will be zero, and we can skip the accumulation
    matmul_cublaslt(dweight, inp, dout, dbias, C, OC, B*T, stream, false, true, 0, 0, 0, 0,
                    global_current_micro_step == 0 ? false : true, NULL, true, false);

    generate_analysis(dweight, (size_t)C*(size_t)OC, "matmul_bwd2_wgrad_dweight");
}
