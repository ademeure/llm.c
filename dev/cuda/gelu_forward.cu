/*
Kernels for gelu forward pass.

Compile example:
nvcc -O3 --use_fast_math -lcublas -lcublasLt gelu_forward.cu -o gelu_forward

If encountering "error: identifier "M_PI" is undefined", add the following lines to the top of the file:

#define _USE_MATH_DEFINES
#include <math.h>  OR  #include <cmath>

version 1 is naive CPU port
./gelu_forward 1

version 2 is bfloat16 with the Packed128 data structure
./gelu_forward 2
*/

size_t B = 1024;

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <assert.h>
#include <vector>
#include <algorithm>

//#define ENABLE_BF16
#define ENABLE_FP32
#include "common.h"

constexpr bool enable_compression = false;
constexpr int num_reads = 16;
constexpr int read_offset = 128;
constexpr int per_clean_iter = 8; // kernel5
constexpr int parallel_blocks = 3; // kernel5

int num_sms, num_threads_per_sm;
unsigned int latency_threshold_far;
unsigned int num_near = 0;
unsigned int num_far = 0;
unsigned int* num_blocks_active;

__constant__ unsigned char c_sm_side[256];
__constant__ unsigned char c_is_far[1024];

constexpr int elements_per_4KiB = 4*1024/sizeof(floatX);
constexpr int elements_per_2MiB = 2*1024*1024/sizeof(floatX);

// ----------------------------------------------------------------------------

__device__ unsigned int latency_of_read(const unsigned char* address) {
    unsigned int clock_start, clock_end;
    asm volatile("mov.u32 %0, %%clock;" : "=r"(clock_start) :: "memory");
    unsigned char value = *address;
    if (value == 255) __nanosleep(1); // avoid dead code optimisation (typically memset to 0)
    asm volatile("mov.u32 %0, %%clock;" : "=r"(clock_end) :: "memory");

    if (clock_end >= clock_start) {
        return clock_end - clock_start;
    } else {
        return (0xFFFFFFFF - clock_start) + clock_end;
    }
}

__global__ void page_side_latency(const unsigned char* data, unsigned int* latencies, unsigned int* smid, size_t num_chunks, size_t granularity) {
    if (threadIdx.x > 0) return;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(*smid));

    for (int i = 0; i < num_chunks; i++) {
        const unsigned char* address = &data[i*granularity];

        #pragma unroll
        for (int k = 0; k < num_reads; k++) {
            unsigned int latency = latency_of_read(address);
            latencies[i*num_reads + k] = latency;
            address += read_offset;
        }
    }
}

__global__ void latency_kernel(const unsigned char* data, unsigned char *block_is_far, unsigned int latency_threshold_far, int N) {
    constexpr size_t num_fetches = 8*2048;
    if (threadIdx.x > 0) return;
    int is_far = 0;
    #pragma unroll 1
    for (int i = 0; i < num_fetches; i++) {
        unsigned int latency = latency_of_read(&data[i*512]);
        is_far += (latency >= latency_threshold_far) ? 1 : 0;
        if ((i % 8) == 7) {
            block_is_far[i/8] = (is_far > 4) ? 1 : 0;
            is_far = 0;
        }
    }
}

__device__ void swap_if_greater(unsigned int& a, unsigned int& b) { // branch-free for GPU sort
    unsigned int old_a = a;
    a = min(a, b);
    b = max(old_a, b);
}

__global__ void sm_kernel(const unsigned char* data, const unsigned char *block_is_far, unsigned char* sm_side) {
    if (threadIdx.x > 0) return; // 1st thread of every block only
    unsigned int smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));

    // get the latency values within two 4KiB chunks that are 2MiB apart
    // 4KiB is the minimum between side changes and 2MiB is the physical page size
    unsigned int latencies[8], latencies_2[8];
    for (int k = 0; k < 8; k++) {
        latencies[k] = latency_of_read(&data[(blockIdx.x*8 + k) * 512]);
        latencies_2[k] = latency_of_read(&data[(blockIdx.x*8 + k) * 512 + 2048*1024]);
    }

    // find the median of latency arrays (-ish, [3] instead of ([3]+[4])/2 to be more conservative)
    // brute force sort that avoids all branches (would not scale well to more than 8 reads...)
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8 - i - 1; j++) {
            swap_if_greater(latencies[j], latencies[j+1]);
            swap_if_greater(latencies_2[j], latencies_2[j+1]);
        }
    }
    unsigned int median_1 = latencies[3];
    unsigned int median_2 = latencies_2[3];

    // far vs near is relative to the arbitrary SM that was used fro the initial latency values
    // we know whether that SM would've had a higher latency (far) for the 1st or 2nd set of reads
    // so if it is the same, this SM is on the same side; othewise, it is on the opposite side
    bool start_far = block_is_far[blockIdx.x];
    sm_side[smid] = (median_1 > median_2) ? start_far : !start_far;
}

void clear_l2() {
    // Get actual L2 size via CUDA on first call of this function
    static int l2_clear_size = 0;
    static unsigned char* gpu_scratch_l2_clear = NULL;
    if (!gpu_scratch_l2_clear) {
        cudaDeviceGetAttribute(&l2_clear_size, cudaDevAttrL2CacheSize, 0);
        l2_clear_size *= 4;
        cudaCheck(cudaMalloc(&gpu_scratch_l2_clear, l2_clear_size));
    }
    // Clear L2 cache (this is run on every call unlike the above code)
    cudaMemset(gpu_scratch_l2_clear, 0, l2_clear_size);
}

// ----------------------------------------------------------------------------
// CUDA memory allocation with compressible memory support

CUmemAllocationProp get_allocation_constraints(size_t &granularity, bool &use_compression)
{
    int compression_available;
    cuDeviceGetAttribute(&compression_available, CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED, 0);
    use_compression = use_compression && compression_available;

    CUmemAllocationProp prop = {};
    memset(&prop, 0, sizeof(CUmemAllocationProp));
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = 0; // force device 0 for now
    prop.allocFlags.compressionType = use_compression ? CU_MEM_ALLOCATION_COMP_GENERIC : 0;

    assert(!cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    assert(granularity == 2048*1024); // todo - currently only support 2MiB granularity
    return prop;
}

void allocateCompressible(void **addr, size_t size, bool use_compression)
{
    //cudaCheck(cudaMalloc(&addr, size); return;

    size_t granularity;
    CUmemAllocationProp prop = get_allocation_constraints(granularity, use_compression);
    size = ((size - 1) / granularity + 1) * granularity;
    size_t num_chunks = size / granularity;

    CUdeviceptr dptr;
    assert(!cuMemAddressReserve(&dptr, size, 0, 0, 0));
    // todo - make sure the virtual address is 4MiB aligned (2x granularity)
    // so that different allocations can match by starting with the same hash

    // allocate each chunk/page of minimum granularity (must be 2MiB for now)
    std::vector <CUmemGenericAllocationHandle> allocationHandles(num_chunks);
    for (size_t i = 0; i < num_chunks; i++) {
        assert(!cuMemCreate(&allocationHandles[i], granularity, &prop, 0));
        if (use_compression) {
            CUmemAllocationProp allocationProp = {};
            assert(!cuMemGetAllocationPropertiesFromHandle(&allocationProp, allocationHandles[i]));
            assert(allocationProp.allocFlags.compressionType == CU_MEM_ALLOCATION_COMP_GENERIC);
        }
    }
    // initial mapping which we'll use to determine the hashing of each page/chunk before remapping
    for (size_t i = 0; i < num_chunks; i++) {
        assert(!cuMemMap(dptr + i * granularity, granularity, 0, allocationHandles[i], 0));
    }
    // make allocation readable & writable (on the virtual memory range, not the physical one)
    CUmemAccessDesc accessDescriptor;
    accessDescriptor.location.id = prop.location.id;
    accessDescriptor.location.type = prop.location.type;
    accessDescriptor.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    assert(!cuMemSetAccess(dptr, size, &accessDescriptor, 1));
    assert(!cuMemsetD8(dptr, 0, size)); // memset allocation to 0
    clear_l2(); // make sure we always miss in the L2 for our allocation

    // alloc latencies array with host cuda malloc
    unsigned int* smid;
    unsigned int* latencies;
    assert(!cuMemHostAlloc((void**)&smid, sizeof(unsigned int), 0));
    assert(!cuMemHostAlloc((void**)&latencies, num_chunks * num_reads * sizeof(unsigned int), 0));

    // launch kernel to measure latency
    page_side_latency<<<1, 32>>>((const unsigned char*)dptr, latencies, smid, num_chunks, granularity);
    cudaCheck(cudaDeviceSynchronize());

    // get the median latency for each chunk and save them for further use
    for (int i = 0; i < num_chunks; i++) {
        std::vector<unsigned int> chunk_latencies;
        for (int n = 0; n < num_reads; n++) {
            chunk_latencies.push_back(latencies[i*num_reads + n]);
        }
        std::sort(chunk_latencies.begin(), chunk_latencies.end());
        latencies[i] = chunk_latencies[num_reads / 2];
    }

    // Get the average of all the medians
    unsigned int total_median_latency = 0;
    for (int i = 0; i < num_chunks; i++) {
        total_median_latency += latencies[i];
    }
    unsigned int average_median_latency = total_median_latency / num_chunks;
    latency_threshold_far = average_median_latency;

    // unmap our initial unoptimised mapping
    for (size_t i = 0; i < num_chunks; i++) {
        assert(!cuMemUnmap(dptr + i * granularity, granularity));
    }

    // Is bit 21 of the virtual address true? If so, start with far allocations
    // todo - force this to always be true (or false or whatever)
    int start_far = (dptr & (1 << 21)) ? 1 : 0;
    assert(start_far == 0);

    // Remap alternating "far" and "near" allocations
    // i.e. if latencies[i] > average_median_latency, then an allocation is far, else near
    int current_near = 0;
    int current_far = 0;
    int i = 0;
    while (i < num_chunks) {
        if ((i % 2) == start_far) {
            while (current_far < num_chunks && latencies[current_far] <= average_median_latency) {
                current_far++;
            }
            if (current_far >= num_chunks) break;
            assert(!cuMemMap(dptr + i * granularity, granularity, 0, allocationHandles[current_far], 0));
            current_far++;
        } else {
            while (current_near < num_chunks && latencies[current_near] > average_median_latency) {
                current_near++;
            }
            if (current_near >= num_chunks) break;
            assert(!cuMemMap(dptr + i * granularity, granularity, 0, allocationHandles[current_near], 0));
            current_near++;
        }
        i++;
    }

    int current = min(current_near, current_far);
    while(i < num_chunks) {
        if ((current_near >= num_chunks && latencies[current] > average_median_latency)
         || (current_far >= num_chunks && latencies[current] <= average_median_latency)) {
            assert(!cuMemMap(dptr + i * granularity, granularity, 0, allocationHandles[current], 0));
            i++;
        }
        current++;
    }

    assert(!cuMemSetAccess(dptr, size, &accessDescriptor, 1));

    cudaCheck(cudaFreeHost(smid));
    cudaCheck(cudaFreeHost(latencies));
    for (size_t i = 0; i < num_chunks; i++) {
        assert(!cuMemRelease(allocationHandles[i]));
    }

    *addr = (void*)dptr;
}

void freeCompressible(void *ptr, size_t size, bool UseCompressibleMemory)
{
    // cudaFree(ptr); return;
    if (ptr == NULL)
        return;
    size_t granularity;
    CUmemAllocationProp prop = get_allocation_constraints(granularity, UseCompressibleMemory);

    size_t chunks = ceil_div(size, granularity);
    for (size_t i = 0; i < chunks; i++) {
        assert(!cuMemUnmap((CUdeviceptr)ptr + i * granularity, granularity));
    }
}

// ----------------------------------------------------------------------------
// CPU code reference

#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)

void gelu_forward_cpu(float* out, const float* inp, int N) {
    for (int i = 0; i < N; i++) {
        float x = inp[i];
        float cube = 0.044715f * x * x * x;
        out[i] = 0.5f * x * (1.0f + tanhf(GELU_SCALING_FACTOR * (x + cube)));
    }
}

// ----------------------------------------------------------------------------
// GPU kernels

__device__ float gelu_forward_element(float xi) {
    float cube = 0.044715f * xi * xi * xi;
    float tanh_in_out = GELU_SCALING_FACTOR * (xi + cube);
    #if !defined(PRECISE_GELU_TANH) && __CUDA_ARCH__ >= 750
    asm ("tanh.approx.f32 %0,%1;" : "=f"(tanh_in_out) : "f"(tanh_in_out));
    #else
    tanh_in_out = tanhf(tanh_in_out);
    #endif

    // the following uses FMUL+FMA instead of FMUL+FADD+FMUL for "0.5f * x * (1.0f + tanh_out)"
    float half_xi = 0.5f * xi;
    return half_xi * tanh_in_out + half_xi;
}

// Optimised with option to use optimised HW TANH instruction by default
__global__ void gelu_forward_kernel3(floatX* out, const floatX* inp, int N) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;
    if (idx >= N) { return; }

    x128 packed_out;
    x128 packed_inp = load128cs(inp + idx); // do not keep in cache
    for(int k = 0; k < packed_inp.size; ++k) {
        packed_out[k] = (floatX)gelu_forward_element((float)packed_inp[k]);
    }
    store128(out + idx, packed_out);
}

__global__ __launch_bounds__(256, parallel_blocks)
void gelu_forward_kernel5(floatX* out, const floatX* inp,
                          int clean_iter_per_block, int plausible_iter_per_block,
                          int num_near, int num_far,
                          unsigned int* num_blocks_active, int N) {
    unsigned int smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    unsigned int blocks_per_sm = parallel_blocks;

    __shared__ unsigned int shared_block_in_sm;
    if (threadIdx.x == 0) {
        shared_block_in_sm  = atomicInc(num_blocks_active + smid, blocks_per_sm-1);
    }

    unsigned int sm_side_raw = c_sm_side[smid];
    unsigned int sm_side_is_near = sm_side_raw & 1;
    unsigned int sm_side_id = sm_side_raw >> 1;
    unsigned int min_per_side = min(num_near, num_far);
    int blocks_per_side = min_per_side * blocks_per_sm;
    int stride = elements_per_4KiB * blocks_per_side;

    __syncthreads();
    unsigned int block_in_sm = shared_block_in_sm;
    int effective_block_id = (sm_side_id * blocks_per_sm) + block_in_sm;
    //assert(block_in_sm < blocks_per_sm);

    if (sm_side_id >= min_per_side) {
        // todo - stragglers do X% + dynamic???
        __nanosleep(20000);
        return;
    }

    unsigned int idx = effective_block_id * elements_per_4KiB + threadIdx.x * x128::size;
    floatX* original_out = out;

    int chunk_id = ((size_t)(inp + idx) >> 12) & 1023;
    idx += (chunk_id >= 512) ? elements_per_2MiB : 0;
    chunk_id &= 511;

    for (int x = 0; x < clean_iter_per_block; x += per_clean_iter) {
        x128 packed_inps[per_clean_iter];
        unsigned int out_offset[per_clean_iter];

        inp += idx;
        out += idx;
        idx = 0;

        for (int y = 0; y < per_clean_iter; y++) {
            unsigned int idx_plus_2MiB = idx + elements_per_2MiB;
            unsigned int idx2 = (c_is_far[chunk_id] == sm_side_is_near) ? idx_plus_2MiB : idx;
            packed_inps[y] = load128cs(inp + idx2);
            out_offset[y] = idx2;

            chunk_id = (chunk_id + blocks_per_side);
            idx += stride + ((chunk_id >= 512) ? elements_per_2MiB : 0);
            chunk_id &= 511;
        }
        __syncthreads();
        __threadfence_block();

        #pragma unroll
        for (int y = 0; y < per_clean_iter; y++) {
            x128 packed_out;
            x128 packed_inp = packed_inps[y];
            for(int k = 0; k < packed_inp.size; ++k) {
                packed_out[k] = (floatX)gelu_forward_element((float)packed_inp[k]);
            }
            store128(out + out_offset[y], packed_out);
        }
    }

    size_t real_idx = (size_t)(out - original_out);

    for(int j = 0; j < plausible_iter_per_block; j++) {
        unsigned int offset = (c_is_far[chunk_id] == sm_side_is_near) ? elements_per_2MiB : 0;

        if (real_idx + offset < N) {
            x128 packed_out;
            x128 packed_inp = load128cs(inp + offset + idx);
            for(int k = 0; k < packed_inp.size; ++k) {
                packed_out[k] = (floatX)gelu_forward_element((float)packed_inp[k]);
            }
            store128(out + offset + idx, packed_out);
        }

        chunk_id += blocks_per_side;
        idx += stride + (chunk_id >= 512 ? elements_per_2MiB : 0);
        chunk_id &= 511;
    }
}




// ----------------------------------------------------------------------------
// kernel launcher

void gelu_forward3(floatX* out, const floatX* inp, int N, const int block_size) {
    const int grid_size = ceil_div(N, block_size * x128::size);
    gelu_forward_kernel3<<<grid_size, block_size>>>(out, inp, N);
    cudaCheck(cudaGetLastError());
}

constexpr float use_wasted_multiplier = 0.0f;

void gelu_forward5(floatX* out, const floatX* inp, int N, int block_size) {
    block_size = 256; // only support 256 for now (and possibly ever)
    int blocks_per_sm = parallel_blocks; // hardcoded for now

    int min_per_side = min(num_near, num_far);
    int num_4kib_chunks = ceil_div(N, elements_per_4KiB);

    int wasted_SMs = num_sms - (2 * min_per_side);
    float percentage_wasted = (float)wasted_SMs / (float)num_sms;
    float adjusted_used = 1.0f - (use_wasted_multiplier * percentage_wasted);
    int num_4kib_chunks_main = (int)ceilf((float)num_4kib_chunks * adjusted_used);

    int effective_num_blocks = 2 * min_per_side * blocks_per_sm;
    int safe_4kib_chunks_main = max(0, num_4kib_chunks_main - 512);
    int clean_iter_per_block = ((safe_4kib_chunks_main / effective_num_blocks) / per_clean_iter) * per_clean_iter;
    int plausible_4kib_chunks_main = ceil_div(num_4kib_chunks_main, 1024) * 1024; // todo - optimise
    int plausible_iter_per_block = (plausible_4kib_chunks_main / effective_num_blocks) - clean_iter_per_block + 1;

    int grid_size = num_sms * blocks_per_sm;
    gelu_forward_kernel5<<<grid_size, block_size>>>(out, inp, clean_iter_per_block, plausible_iter_per_block,
                                                    num_near, num_far,
                                                    num_blocks_active, N);
    cudaCheck(cudaGetLastError());
}


// kernel version dispatch
void gelu_forward(int kernel_num,
                  floatX* out,
                  const floatX* inp,
                  int B, int T, int C,
                  int block_size) {
    switch (kernel_num) {
        case 3:
            gelu_forward3(out, inp, B * T * C, block_size);
            break;
        case 5:
            gelu_forward5(out, inp, B * T * C, block_size);
            break;
        default:
            printf("Invalid kernel number\n");
            exit(1);
    }
}

// ----------------------------------------------------------------------------

int main(int argc, const char **argv) {
    setup_main();

    size_t T = 1024;
    size_t C = 768;

    // create host memory of random numbers
    float* out = (float*)malloc(B * T * C * sizeof(float));
    float* inp = make_random_float(B * T * C);

    // read kernel_num from command line
    int kernel_num = (argc > 1) ? atoi(argv[1]) : 1;
    printf("Using kernel %d\n", kernel_num);

    // first check the correctness of the kernel
    gelu_forward_cpu(out, inp, B * T * C);

    cudaCheck(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0));
    cudaCheck(cudaDeviceGetAttribute(&num_threads_per_sm, cudaDevAttrMaxThreadsPerMultiProcessor, 0));

    unsigned char *sm_side, *block_is_far;
    cudaCheck(cudaMallocHost((void**)&block_is_far, 2048 * sizeof(unsigned char)));
    cudaCheck(cudaMallocHost((void**)&sm_side, num_sms * sizeof(unsigned char)));

    cudaCheck(cudaMalloc((void**)&num_blocks_active, num_sms * 64 * sizeof(unsigned int)));
    cudaMemset(num_blocks_active, 0, num_sms * 64 * sizeof(unsigned int));

    // GPU allocations
    floatX *d_out, *d_inp;
    allocateCompressible((void**)&d_out, B * T * C * sizeof(floatX) * 2, enable_compression);
    allocateCompressible((void**)&d_inp, B * T * C * sizeof(floatX) * 2, enable_compression);

    // Figure out which 4KiB chunks are far and which are near (for a single arbitrary SM)
    clear_l2();
    latency_kernel<<<1, 32>>>((unsigned char*)d_out, block_is_far, latency_threshold_far, B * T * C);

    // Let's figure out which SMs are on the "near" and "far" sides (relative to the arbitrary SM)
    clear_l2();
    sm_kernel<<<num_sms, 512>>>((unsigned char*)d_out, block_is_far, sm_side);
    cudaCheck(cudaDeviceSynchronize());

    for (int i = 0; i < num_sms; i++) {
        if (sm_side[i]) {
            sm_side[i] |= (num_far++) << 1;
        } else {
            sm_side[i] |= (num_near++) << 1;
        }
    }

    // move data & constants to GPU
    cudaCheck(memcpy_convert(d_inp, inp, B * T * C));
    cudaMemcpyToSymbol(c_sm_side, sm_side, num_sms * sizeof(unsigned char));
    cudaMemcpyToSymbol(c_is_far, block_is_far, 1024 * sizeof(unsigned char));
    cudaCheck(cudaDeviceSynchronize());

    // time & validate the kernel at different block sizes
    int block_sizes[] = {256};
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        printf("Checking block size %d.\n", block_sizes[j]);
        gelu_forward(kernel_num, d_out, d_inp, B, T, C, block_sizes[j]);
        float tol = sizeof(floatX) >= sizeof(float) ? 1e-5f : 1e-3f;
        validate_result(d_out, out, "out", B * T * C, tol);
        cudaMemset(d_out, 0, B * T * C * sizeof(floatX)); // make sure results aren't from last run
    }
    printf("All results match. Starting benchmarks.\n\n");

    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];

        int repeat_times = 1000;
        float elapsed_time = benchmark_kernel(repeat_times, gelu_forward,
                                              kernel_num, d_out, d_inp,
                                              B, T, C, block_sizes[j]);

        size_t memory_ops = B * T * C * 2 * sizeof(floatX); // 1 read + 1 write
        float memory_bandwidth = memory_ops / elapsed_time / 1e6;
        printf("block_size %4d | time %.4f ms | bandwidth %.2f GB/s\n", block_size, elapsed_time, memory_bandwidth);
    }

    // free memory
    freeCompressible(d_out, B * T * C * sizeof(floatX), enable_compression);
    freeCompressible(d_inp, B * T * C * sizeof(floatX), enable_compression);
    free(out);
    free(inp);
    return 0;
}