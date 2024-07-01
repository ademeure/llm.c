/*
Matrix Multiplication, with help from cuBLASLt
*/
#include <assert.h>
#include <type_traits>      // std::bool_constant
// llmc internal imports
#include "cuda_common.h"
#include "cuda_utils.cuh"
#include "cublas_common.h"
// GELU can be either fused (cublasLt) or non-fused (gelu.h)
#include "gelu.cuh"
#include <cuda_fp8.h>

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

// ============================================================================
// ABSMAX & FRIENDS
// ============================================================================

#define ABSMAX_OUTER_LOOP 4

typedef Packed128<__nv_fp8_e4m3> e4m3_128;

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
            // IMPORTANT: that would be non-deterministic due to denormals (and maybe other factors)
            // but it'd increase performance by reducing the probability we need to do a second pass
            //absmax_uint = max(absmax_uint, atomicMax((unsigned int*)absmax_scalar, absmax_uint));

            atomicMax((unsigned int*)absmax_scalar, absmax_uint);
            tmp[0] = absmax_uint;
        }
    }
    __syncthreads();
    float absmax = __uint_as_float(tmp[0]);
    floatX scale = (floatX)(absmax != 0.0f ? (1.0f / absmax) : 1.0f);

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
__global__ void __launch_bounds__(512, 4) rescale_kernel(__nv_fp8_e4m3* in_out, float* absmax_scalar, unsigned char* absmax_vector, size_t N) {
    float absmax = __uint_as_float(*((unsigned int*)absmax_scalar));
    floatX scale = (floatX)(absmax != 0.0f ? (1.0f / absmax) : 1.0f);

    unsigned char target_exponent = (unsigned char)((*(unsigned short*)&scale) >> 7);
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
    float absmax = __uint_as_float(*((unsigned int*)absmax_scalar));
    floatX scale = (floatX)(absmax != 0.0f ? (1.0f / absmax) : 1.0f);

    unsigned char target_exponent = (unsigned char)((*(unsigned short*)&scale) >> 7);
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

        size_t idx = (((blockIdx.x * consecutive + i) * blockDim.x) + threadIdx.x) * Packed128<__nv_fp8_e4m3>::size;
        int exponent_difference = target_exponent - current_exponent;
        float scale_factor = exp2f((float)exponent_difference);

        Packed128<__nv_fp8_e4m3> packed_data = load128(in_out + idx);
        for (int k = 0; k < Packed128<__nv_fp8_e4m3>::size; k++) {
            float x = (float)packed_data[k];
            x *= scale_factor;
            packed_data[k] = (__nv_fp8_e4m3)x;
        }
        store128(in_out + idx, packed_data);
    }
}

template <bool scaled=false, typename T1, typename T2>
__global__ void copy128_kernel(T1 *out, const T2 *inp, size_t N, float scale_factor) {
    Packed128<T1> out128;
    size_t n = (blockIdx.x * blockDim.x + threadIdx.x) * out128.size;
    if (n >= N) { return; }

    // note: if sizeof(T1) < sizeof(T2), compiler will skip unused elements of load128
    // so it may turn out to be a load32 or load64
    Packed128<T2> inp128;
    for (int o = 0; o < max(1UL, out128.size/inp128.size); o++) {
        inp128 = load128<T2>(inp + n + o*inp128.size);
        for (int k = 0; k < min(inp128.size, out128.size); k++) {
            if constexpr (scaled) {
                out128[k+o*inp128.size] = (T1)((float)inp128[k] * scale_factor);
            } else {
                out128[k+o*inp128.size] = (T1)inp128[k];
            }
        }
    }
    store128<T1>(out + n, out128);
}

template <typename T1, typename T2>
void copy128(T1* out, const T2* inp, size_t N, float scale_factor=1.0f) {
    constexpr size_t block_size = 64;
    size_t elements_per_block = block_size * 16 / sizeof(T1);
    assert(N % elements_per_block == 0);

    if (scale_factor != 1.0f) {
        copy128_kernel<true><<<dim3(N / elements_per_block), dim3(block_size)>>>(out, inp, N, scale_factor);
    } else {
        copy128_kernel<false><<<dim3(N / elements_per_block), dim3(block_size)>>>(out, inp, N, 1.0f);
    }
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
    stream_attribute.accessPolicyWindow.missProp  = cudaAccessPropertyNormal;
    cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);
    */

    if (grid_size > deviceProp.multiProcessorCount * 20 && (grid_size % 8) == 0) {
        rescale_kernel_multi<8><<<grid_size/8, 512, 0, stream>>>(out, absmax_global, absmax_vector, N);
    } else {
        rescale_kernel<<<grid_size, 512, 0, stream>>>(out, absmax_global, absmax_vector, N);
    }
    cudaCheck(cudaGetLastError());
}

// ----------------------------------------------------------------------------
// kernel launchers

#define FORCE_FP8_MATMUL
float initial_absmax[2] = {1.0f, 1.0f};

// Wrapper around cublasLtMatmul that is meant to support everything we need in llm.c
// https://docs.nvidia.com/cuda/cublas/#cublasltmatmul
void matmul_cublaslt(floatX* d, floatX* a, floatX* b, const floatX* bias,
                     int m, int n, int k, cudaStream_t stream=0, bool transA=true, bool transB=false,
                     int batch_count=0, size_t strideA=0, size_t strideB=0, size_t strideOut=0,
                     bool accumulate=false, floatX* pre_gelu=NULL, bool backward=false, bool allow_fp8=true)
{
    NVTX_RANGE_FN();
    bool has_bias = (bias != NULL);
    bool has_gelu = (pre_gelu != NULL);

    // check alignment (some modes work unaligned but it always best to be aligned for performance)
    if(((uintptr_t)a % 16) != 0 || ((uintptr_t)b % 16) != 0 || ((uintptr_t)d % 16) != 0 || ((uintptr_t)bias % 16) != 0) {
        printf("All cuBLASLt pointers must be aligned!\n");
        exit(EXIT_FAILURE);
    }

    // create the operation descriptor
    cublasLtMatmulDesc_t operationDesc;
    cublasCheck(cublasLtMatmulDescCreate(&operationDesc, cublas_compute, CUDA_R_32F));

    int returnedResults = 0;
    cublasLtMatmulPreference_t preference;
    cublasLtMatmulHeuristicResult_t heuristic;

    cublasOperation_t opNoTranspose = CUBLAS_OP_N;
    cublasOperation_t opTranspose = CUBLAS_OP_T;
    cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, (transA)  ? &opTranspose : &opNoTranspose,   sizeof(opTranspose)));
    cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, (transB) ? &opTranspose   : &opNoTranspose, sizeof(opNoTranspose)));

    // todo - hack - this memory shouldn't be allocated here, obviously...
    static void* huge_scratch = NULL;
    static size_t huge_scratch_size = 0;
    size_t needed_size = 128 + 2 * max(m*k, n*k) * sizeof(float) + 2 * max(m*k, n*k) * sizeof(float); // todo - wrong, too big
    if (huge_scratch_size < needed_size) {
        if (huge_scratch_size > 0) {
            cudaCheck(cudaFree(huge_scratch));
        }
        // Allocate a huge scratch space for FP8 mode
        huge_scratch_size = needed_size;
        cudaCheck(cudaMalloc(&huge_scratch, huge_scratch_size));
    }

    auto input_precision = CUBLAS_LOWP;
    #ifdef FORCE_FP8_MATMUL
    if (transA && !transB && batch_count == 0 && allow_fp8) {
        input_precision = CUDA_R_8F_E4M3;
        int8_t fast_accum = 1;
        cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fast_accum, sizeof(fast_accum)));

        // Get absmax for a and b
        size_t absmax_size = CEIL_DIV(8*max(n*k,m*k), 256 * 8 * x128::size);
        size_t rounded_to_128a = CEIL_DIV(128+absmax_size, 128) * 128;
        size_t rounded_to_128b = CEIL_DIV(128+absmax_size+m*k, 128) * 128;

        unsigned char* absmax_vector = &((unsigned char*)huge_scratch)[128];
        float* absmax_a = &((float*)huge_scratch)[0];
        float* absmax_b = &((float*)huge_scratch)[1];
        __nv_fp8_e4m3* a_fp8 = &((__nv_fp8_e4m3*)huge_scratch)[rounded_to_128a];
        __nv_fp8_e4m3* b_fp8 = &((__nv_fp8_e4m3*)huge_scratch)[rounded_to_128b];

        cudaMemcpyAsync(absmax_a, initial_absmax, 2 * sizeof(float), cudaMemcpyHostToDevice, stream);
        //get_absmax_and_scale(a_fp8, a, absmax_a, absmax_vector, false, m*k, 1.0f/128.0f, stream);
        //get_absmax_and_scale(b_fp8, b, absmax_b, absmax_vector, false, n*k, 1.0f/128.0f, stream);

        a = (floatX*)a_fp8;
        b = (floatX*)b_fp8;

        // Set CUBLASLT_MATMUL_DESC_A_SCALE_POINTER
        cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &absmax_a, sizeof(absmax_a)));
        cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &absmax_b, sizeof(absmax_b)));
    }
    #endif

    // define matrix layouts
    cublasLtMatrixLayout_t ALayout;
    cublasLtMatrixLayout_t BLayout;
    cublasLtMatrixLayout_t DLayout;
    cublasLtMatrixLayout_t CLayout;
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
    // cuBLASLt requires C in FP8 mode to be BF16 or FP32... (sigh)
    cublasCheck(cublasLtMatrixLayoutCreate(&CLayout, (sizeof(floatX) == 1) ? CUDA_R_16BF : CUBLAS_LOWP, m, n, m));
    cublasCheck(cublasLtMatrixLayoutCreate(&DLayout, CUBLAS_LOWP, m, n, m));

    // Strided Batched GEMM (used for non-flash attention, equivalent to cublasGemmStridedBatchedEx)
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
    cublasCheck(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                     &cublaslt_workspace_size, sizeof(cublaslt_workspace_size)));

    // setup epilogue and associated pointers for bias & gelu
    cublasLtEpilogue_t epilogue;
    if (has_gelu) {
        int64_t gelu_ld = m; // todo - is this affected by anything else?
        cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, &gelu_ld, sizeof(gelu_ld)));
        cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, &pre_gelu, sizeof(pre_gelu)));
        if (backward) {
            assert(!has_bias); // we shouldn't have any backward matmuls that use both GELU and bias
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
        // cuBLASLt requires bias in FP8 mode to be BF16... (sigh)
        cublasDataType_t bias_data_type = (sizeof(floatX) == 1) ? CUDA_R_16BF : CUBLAS_LOWP; // force BF16 bias for FP8 mode
        cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_data_type, sizeof(bias_data_type)));
        cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
    }

    // set scale type to FP32 (needs to be FP16 if and only if using CUBLAS_COMPUTE_16F, so it's FP32 even for FP8!)
    cublasDataType_t scale_type = CUDA_R_32F;
    cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_SCALE_TYPE, &scale_type, sizeof(scale_type)));

    // find a suitable algorithm (cached internally so shouldn't take much CPU time in practice)
    cublasLtMatmulAlgoGetHeuristic(cublaslt_handle, operationDesc, ALayout, BLayout, CLayout, DLayout,
                                   preference, 1, &heuristic, &returnedResults);
    if (returnedResults == 0) {
        printf("No cuBLASLt algorithm: m: %d, n: %d, k: %d, bias: %d\n", n, m, k, has_bias);
        exit(EXIT_FAILURE);
    }

    // set whether to accumulate (i.e. D += C) or not - note this isn't considered in algorithm selection (?!)
    const float alpha = 1.0f, beta = accumulate ? 1.0f : 0.0f;

    // call the matmul
    cublasCheck(cublasLtMatmul(cublaslt_handle, operationDesc,
                               &alpha, a, ALayout, b, BLayout, &beta, d, CLayout, d, DLayout,
                               &heuristic.algo, cublaslt_workspace, cublaslt_workspace_size, stream));

    // cleanups
    cublasCheck(cublasLtMatmulPreferenceDestroy(preference));
    cublasCheck(cublasLtMatmulDescDestroy(operationDesc));
    cublasCheck(cublasLtMatrixLayoutDestroy(ALayout));
    cublasCheck(cublasLtMatrixLayoutDestroy(BLayout));
    cublasCheck(cublasLtMatrixLayoutDestroy(CLayout));
    cublasCheck(cublasLtMatrixLayoutDestroy(DLayout));
    cudaCheck(cudaGetLastError());
}

// small wrapper around matmul_cublaslt for the forward pass (keeping historical order of arguments)
void matmul_forward_cublaslt(floatX* out,
                     floatX* inp, floatX* weight, floatX* bias,
                     int B, int T, int C, int OC, cudaStream_t stream,
                     floatX* pre_gelu=NULL, int gelu_fusion=1) {
    // By default only fuse GELU for H100+ as cuBLAS seems to be inefficient for fused GELU on Ada/Ampere (?)
    if (gelu_fusion < 1 && pre_gelu) {
        matmul_cublaslt(pre_gelu, weight, inp, bias, OC, B*T, C, stream, true, false, 0, 0, 0, 0, false, NULL, false);
        gelu_forward(out, pre_gelu, B*T*OC, stream);
    } else {
        matmul_cublaslt(out, weight, inp, bias, OC, B*T, C, stream, true, false, 0, 0, 0, 0, false, pre_gelu, false);
    }
}

void matmul_backward(floatX* dinp, floatX* dweight, floatX* dbias,
                     floatX* dout, floatX* inp, floatX* weight,
                     float* dbias_buffer,
                     int B, int T, int C, int OC, cudaStream_t stream,
                     floatX* pre_gelu=NULL, int gelu_fusion=1) {
    NVTX_RANGE_FN();

    // backward to bias, if given, does a +=
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
            matmul_backward_bias_kernel9<<<dim3(grid_size_x, grid_size_y), block_dim, 0, stream>>>(dbias, dout, B, T, OC, False);
            cudaCheck(cudaGetLastError());
        } else {
            // kernel 9 overwrites temp buffer, so no need to memset
            matmul_backward_bias_kernel9<<<dim3(grid_size_x, grid_size_y), block_dim, 0, stream>>>(dbias_buffer, dout, B, T, OC, True);
            cudaCheck(cudaGetLastError());
            reduce_add_sum_kernel<<<CEIL_DIV(OC, 256 * f128::size), 256, 0, stream>>>(dbias, dbias_buffer, OC, grid_size_y);
            cudaCheck(cudaGetLastError());
        }
        dbias = NULL; // prevent dbias calculation from also being fused in matmul_cublaslt below (if we enabled fusion)
    }

    // backward to input, uses = in the backward pass (set the gradient)
    matmul_cublaslt(dinp, weight, dout, NULL, C, B*T, OC, stream, false, false, 0, 0, 0, 0, false,
                    gelu_fusion >= 2 ? pre_gelu : NULL, true);

    // backward GELU (if it wasn't fused into the matmul above)
    if (gelu_fusion < 2 && pre_gelu) {
        gelu_backward_inplace(dinp, pre_gelu, B*T*C, stream);
    }

    // backward to weight, uses += in the backward pass (accumulate the gradient) by setting alpha=one
    matmul_cublaslt(dweight, inp, dout, NULL /*dbias*/, C, OC, B*T, stream, false, true, 0, 0, 0, 0,
                    true /* accumulate */, NULL, true);
}
