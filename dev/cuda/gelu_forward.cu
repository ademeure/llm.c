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

//#define ENABLE_BF16
#define ENABLE_FP32
#include "common.h"

constexpr unsigned int latency_threshold_side = 570;
constexpr int num_reads = 16;
constexpr int read_offset = 128;
constexpr bool enable_compression = false;

int num_sms;
int num_threads_per_sm;
int num_near = 0;
int num_far = 0;
unsigned char* sm_side;
unsigned char *block_is_far;

constexpr size_t l2_clear_size = 128 * 1024 * 1024;
unsigned char* gpu_scratch_l2_clear;
unsigned int* num_blocks_active;

__constant__ unsigned char c_sm_side[256];
__constant__ unsigned char c_is_far[1024];
__constant__ unsigned char c_to_next[1024];

__global__ void page_side_latency(const unsigned char* data, unsigned int* latencies, unsigned int* smid, size_t num_chunks, size_t granularity) {
    if (threadIdx.x > 0) return;

    unsigned int smid_value;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid_value));
    *smid = smid_value;

    for (int i = 0; i < num_chunks; i++) {
        const unsigned char* address = &data[i*granularity];
        unsigned int clock_start, clock_end;

        #pragma unroll num_reads
        for (int k = 0; k < num_reads; k++) {
            asm volatile("mov.u32 %0, %%clock;" : "=r"(clock_start) :: "memory");
            unsigned char value = *address;
            if (value != 0) return; // dummy exit to avoid dead code optimisation (memset to always be 0)
            asm volatile("mov.u32 %0, %%clock;" : "=r"(clock_end) :: "memory");

            unsigned int latency = (clock_end < clock_start) ? ((0xFFFFFFFF - clock_start) + clock_end) : (clock_end - clock_start);
            latencies[i*num_reads + k] = latency;
            address += read_offset;
        }
    }
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
    // todo - hack required to avoid initialisation issues (?)
    cudaCheck(cudaSetDevice(0));

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
    // memset allocation to 0
    assert(!cuMemsetD8(dptr, 0, size));
    // memset a buffer larger than L2 to evict our allocation from L2 post-memset
    cudaMemset(gpu_scratch_l2_clear, 0, l2_clear_size);
    // alloc latencies array with host cuda malloc
    unsigned int* smid;
    unsigned int* latencies;
    assert(!cuMemHostAlloc((void**)&smid, sizeof(unsigned int), 0));
    assert(!cuMemHostAlloc((void**)&latencies, num_chunks * num_reads * sizeof(unsigned int), 0));
    // launch kernel to measure latency
    page_side_latency<<<1, 32>>>((const unsigned char*)dptr, latencies, smid, num_chunks, granularity);
    cudaCheck(cudaDeviceSynchronize());
    // print all the latencies
    printf("num_chunks: %lu (smid: %u)\n", num_chunks, *smid);
    for (int i = 0; i < num_chunks; i++) {
        for (int n = 0; n < num_reads; n++) {
            printf("%d[%d]: %d\n", i, n, latencies[i*num_reads + n]);
        }
    }
    // unmap our initial unoptimised mapping
    for (size_t i = 0; i < num_chunks; i++) {
        assert(!cuMemUnmap(dptr + i * granularity, granularity));
    }

    // Is bit 21 of the virtual address true? If so, start with far allocations
    // todo - force this to always be true (or false or whatever)
    int start_far = (dptr & (1 << 21)) ? 1 : 0;
    printf("Start far: %d\n", start_far);

    // Remap alternating "far" and "near" allocations
    // i.e. if latencies[i] > latency_threshold_side, then far, else near
    int current_near = 0;
    int current_far = 0;
    int i = 0;
    while (i < num_chunks) {
        if ((i % 2) == start_far) {
            while (current_far < num_chunks && latencies[current_far] <= latency_threshold_side) {
                current_far++;
            }
            if (current_far >= num_chunks) break;
            assert(!cuMemMap(dptr + i * granularity, granularity, 0, allocationHandles[current_far], 0));
            current_far++;
        } else {
            while (current_near < num_chunks && latencies[current_near] > latency_threshold_side) {
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
        if ((current_near >= num_chunks && latencies[current] > latency_threshold_side)
         || (current_far >= num_chunks && latencies[current] <= latency_threshold_side)) {
            assert(!cuMemMap(dptr + i * granularity, granularity, 0, allocationHandles[current], 0));
            i++;
        }
        current++;
    }

    assert(!cuMemSetAccess(dptr, size, &accessDescriptor, 1));
    assert(!cuMemsetD8(dptr, 0, size));
    cudaMemset(gpu_scratch_l2_clear, 0, l2_clear_size);

    // Launch kernel to measure latency
    printf("REMAPPED!!!\n");
    page_side_latency<<<1, 32>>>((const unsigned char*)dptr, latencies, smid, num_chunks, granularity);
    cudaCheck(cudaDeviceSynchronize());

    // Print all the latencies
    printf("(post-remap) num_chunks: %lu (smid: %u)\n", num_chunks, *smid);
    for (int i = 0; i < num_chunks; i++) {
        for (int n = 0; n < num_reads; n++) {
            printf("%d[%d]: %d\n", i, n, latencies[i*num_reads + n]);
        }
    }

    cudaCheck(cudaFreeHost(smid));
    cudaCheck(cudaFreeHost(latencies));
    printf("post-free\n");

    // Release handles
    for (size_t i = 0; i < num_chunks; i++) {
        assert(!cuMemRelease(allocationHandles[i]));
    }

    printf("post-release\n");
    *addr = (void*)dptr;
}

void freeCompressible(void *ptr, size_t size, bool UseCompressibleMemory)
{
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

// elementwise ops are nice and ez
__global__ void gelu_forward_kernel1(floatX* out, const floatX* inp, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float xi = inp[i];
        float cube = 0.044715f * xi * xi * xi;
        out[i] = 0.5f * xi * (1.0f + tanhf(GELU_SCALING_FACTOR * (xi + cube)));
    }
}

// elementwise ops are nice and ez
__global__ void gelu_forward_kernel2(floatX* out, const floatX* inp, int N) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;
    if (i < N) {
        x128 packed_out;
        x128 packed_inp = load128cs(inp + i); // load and do not keep in cache
        for(int k = 0; k < packed_inp.size; ++k) {
            float xi = (float)packed_inp[k];
            float cube = 0.044715f * xi * xi * xi;
            packed_out[k] = (floatX)(0.5f * xi * (1.0f + tanhf(GELU_SCALING_FACTOR * (xi + cube))));
        }
        // store instead of storecs (without cache streaming) in case it is useful for the
        // data to be in the cache for the next operation after this GeLU
        store128(out + i, packed_out);
    }
}

// Optimised with option to use optimised HW TANH instruction by default
__global__ void gelu_forward_kernel3(floatX* out, const floatX* inp, int N) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;
    if (idx >= N) { return; }

    x128 packed_out;
    x128 packed_inp = load128cs(inp + idx); // load and do not keep in cache
    for(int k = 0; k < packed_inp.size; ++k) {
        float xi = (float)packed_inp[k];
        float cube = 0.044715f * xi * xi * xi;

        float tanh_in_out = GELU_SCALING_FACTOR * (xi + cube);
        #if !defined(PRECISE_GELU_TANH) && __CUDA_ARCH__ >= 750
        asm ("tanh.approx.f32 %0,%1;" : "=f"(tanh_in_out) : "f"(tanh_in_out));
        #else
        tanh_in_out = tanhf(tanh_in_out);
        #endif

        // the following uses FMUL+FMA instead of FMUL+FADD+FMUL for "0.5f * x * (1.0f + tanh_out)"
        float half_xi = 0.5f * xi;
        packed_out[k] = (floatX)(half_xi * tanh_in_out + half_xi);
    }
    // store instead of storecs (without cache streaming) in case it is useful for the
    // data to be in the cache for the next operation after this GeLU
    store128(out + idx, packed_out);
}

__global__ void latency_kernel(const floatX* inp, unsigned char *block_is_far, int N) {
    constexpr size_t num_fetches = 8*2048;
    // large shared memory array
    //__shared__ unsigned int shared[num_fetches];
    if (threadIdx.x > 0) return;
    // print sizeof floatX
    printf("sizeof(floatX): %lu\n", sizeof(floatX));
    int is_far = 0;
    int consecutive_far = 0;
    int consecutive_near = 0;
    int last_change = 0;
    unsigned int last_elapsed_clocks = 0;
    #pragma unroll 1
    for (int n = 0; n < 1; n++) {
        for (int i = 0; i < num_fetches+1; i++) {
            // get clock
            unsigned int clock;
            asm volatile("mov.u32 %0, %%clock;" : "=r"(clock));
            // load inp[x] bypassing L2 using ld.cv: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cache-operators
            floatX x = __ldcv(&inp[i*128]);
            if ((float)x == 99999.999f) {
                return;
            }
            // get clock
            unsigned int clock2;
            asm volatile("mov.u32 %0, %%clock;" : "=r"(clock2));
            // calculate elpased clock cycles
            unsigned int elapsed_clocks = clock2 - clock;
            if ((i % 8) == 0 && i > 0) {
                int old_i_div_8 = (i / 8) - 2;
                int new_i_div_8 = (i / 8) - 1;
                int bit_xor = old_i_div_8 ^ new_i_div_8;
                /*if (is_far > 4) {
                    if (consecutive_near > 0) {
                        //if (consecutive_near != 2 && bit_xor > 32) {
                            //if (new_i_div_8 - last_change != 128) {
                                printf("n: -- %d -- (%d => %d: %d", consecutive_near, old_i_div_8, new_i_div_8, bit_xor);
                                // which bits toggled?
                                for (int j = 7; j >= 0; j--) {
                                    //printf("%d", bit_xor & (1 << j) ? 1 : 0);
                                }
                                printf(" -----> %d)\n", last_change + 128);
                            //}
                            last_change = new_i_div_8;
                        //}
                        consecutive_near = 0;
                    }
                    consecutive_far++;
                } else {
                    if (consecutive_far > 0) {
                       // if (consecutive_far != 2 && bit_xor > 32) {
                            //if (new_i_div_8 - last_change != 128) {
                                printf("F: -- %d -- (%d => %d: %d", consecutive_far, old_i_div_8, new_i_div_8, bit_xor);
                                for (int j = 7; j >= 0; j--) {
                                    //printf("%d", bit_xor & (1 << j) ? 1 : 0);
                                }
                                printf(" -----> %d)\n", last_change + 128);
                            //}
                            last_change = new_i_div_8;
                        //}
                        consecutive_far = 0;
                    }
                    consecutive_near++;
                }
                */
                //printf("%d[%d/%d]: %c (%d)\n", (i/8)-1, (i-8)%4096, (i-8)/4096, is_far > 4 ? 'F' : 'T', last_elapsed_clocks);
                block_is_far[(i/8)-1] = is_far > 4 ? 1 : 0;
                is_far = 0;
                //printf("\n%d,", i/8);
                // print binary representation of first 8 bits of (i/8), e.g. ",00000101" for i=40
                for (int j = 7; j >= 0; j--) {
                    //printf("%d", (i/8) & (1 << j) ? 1 : 0);
                }
            }
            is_far += (elapsed_clocks > latency_threshold_side) ? 1 : 0;
            last_elapsed_clocks = elapsed_clocks;
            //printf(",%d", elapsed_clocks > 600 ? 1 : 0);
            // store in shared memory
            //shared[i] = elapsed_clocks;
            __syncthreads();
        }
        printf("\n======\n");
    }
    // print all the elapsed cycle counts in shared memory
    if (threadIdx.x == 0) {
        for (int i = 0; i < num_fetches; i++) {
            //printf("%d: %u\n", i, shared[i]);
        }
    }
}

__global__ void sm_kernel(const floatX* inp, unsigned char *block_is_far, unsigned char* sm_side) {
    if (threadIdx.x > 0) return;
    int is_far = 0;

    #pragma unroll 1
    for (int k = 0; k < 8; k++) {
        int i = blockIdx.x * 8 + k;
        // get clock
        unsigned int clock;
        asm volatile("mov.u32 %0, %%clock;" : "=r"(clock));
        // load inp[x] bypassing L2 using ld.cv: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cache-operators
        floatX x = __ldcv(&inp[i*128]);
        if ((float)x == 99999.999f) {
            return;
        }
        // get clock
        unsigned int clock2;
        asm volatile("mov.u32 %0, %%clock;" : "=r"(clock2));
        // calculate elpased clock cycles
        unsigned int elapsed_clocks = clock2 - clock;
        is_far += (elapsed_clocks > latency_threshold_side) ? 1 : 0;
    }

    int old_far = block_is_far[blockIdx.x];
    int new_far = is_far > 4 ? 1 : 0;

    // Get SM id
    unsigned int smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));

    sm_side[smid] = (old_far == new_far);
    //printf("SM %d: %d/%d ==> %d\n", smid, old_far, new_far, sm_side[smid]);
}

__global__ void gelu_forward_kernel4a(floatX* out, const floatX* inp, int iter_per_block, int elements_per_block, int num_near, int num_far, int N) {
    int idx = (blockIdx.x * elements_per_block) + (threadIdx.x * x128::size);
    for (int j = 0; j < iter_per_block; j++) {
        if (idx >= N) { return; }
        {
            x128 packed_out;
            x128 packed_inp = load128cs(inp + idx); // load and do not keep in cache
            for(int k = 0; k < packed_inp.size; ++k) {
                float xi = (float)packed_inp[k];
                float cube = 0.044715f * xi * xi * xi;

                float tanh_in_out = GELU_SCALING_FACTOR * (xi + cube);
                #if !defined(PRECISE_GELU_TANH) && __CUDA_ARCH__ >= 750
                asm ("tanh.approx.f32 %0,%1;" : "=f"(tanh_in_out) : "f"(tanh_in_out));
                #else
                tanh_in_out = tanhf(tanh_in_out);
                #endif

                // the following uses FMUL+FMA instead of FMUL+FADD+FMUL for "0.5f * x * (1.0f + tanh_out)"
                float half_xi = 0.5f * xi;
                packed_out[k] = (floatX)(half_xi * tanh_in_out + half_xi);
            }
            // store instead of storecs (without cache streaming) in case it is useful for the
            // data to be in the cache for the next operation after this GeLU
            store128(out + idx, packed_out);
        }
        idx += blockDim.x * x128::size;
    }
}



// Optimised with option to use optimised HW TANH instruction by default
__global__ __launch_bounds__(256, 4)
void gelu_forward_kernel4b(floatX* __restrict__ out, const floatX* __restrict__ inp,
                          int iter_per_block, int elements_per_block, int num_near, int num_far,
                          int inner_loop_iter, unsigned int* __restrict__ num_blocks_active, int N) {
    // get SM id
    unsigned int smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    unsigned int sm_side_raw = c_sm_side[smid];
    unsigned int sm_side_near = sm_side_raw & 1;
    unsigned int sm_side_id = sm_side_raw >> 1;
    unsigned int blocks_per_sm = 4; //(2048 / blockDim.x);

    __shared__ unsigned int shared_block_in_sm;
    if (threadIdx.x == 0) {
        shared_block_in_sm  = atomicInc(num_blocks_active + smid, blocks_per_sm-1);
    }
    __syncthreads();
    unsigned int block_in_sm = shared_block_in_sm;

    int min_per_side = min(num_near, num_far);
    if (sm_side_id >= min_per_side) {
        // todo - stragglers do X% + dynamic???
        // use nanosleep to make sure all the other thread blocks have launched
        // and this won't result in this SM getting more than its expected number of thread blocks
        __nanosleep(10000);
        return;
    }
    int effective_block_id = (sm_side_id * blocks_per_sm) + block_in_sm;

    size_t idx = (effective_block_id * elements_per_block) + (threadIdx.x * x128::size);

    //if (smid != 43 || block_in_sm != 0 || threadIdx.x >= 32) {
    //    return;
    //}

    int j = 0;
    size_t addr = (size_t)(inp + idx);
    int chunk_id = (addr >> 12) & 1023;
    unsigned char is_far = c_is_far[chunk_id];

    /*size_t write_addr = (size_t)(out + idx);
    int write_chunk_id = (write_addr >> 12) & 1023;
    int write_is_far = c_is_far[write_chunk_id];
    assert(write_chunk_id == chunk_id);*/

    //while (is_far != (sm_side_near ^ (effective_block_id & 1))) {
    //while (is_far != sm_side_near) {
    while (is_far == sm_side_near) {
        idx += blockDim.x * x128::size;
        chunk_id = (chunk_id + 1) & 1023;
        is_far = c_is_far[chunk_id];
        j++;
    }
    int clean_iters = (iter_per_block - 8) / 32;

    #pragma unroll 1
    for (int x = 0; x < clean_iters; x++) {
        #pragma unroll
        for (int y = 0; y < 16; y++) {
            x128 packed_out;
            x128 packed_inp = load128cs(inp + idx); // load and do not keep in cache
            #pragma unroll
            for(int k = 0; k < packed_inp.size; ++k) {
                float xi = (float)packed_inp[k];
                float cube = 0.044715f * xi * xi * xi;

                float tanh_in_out = GELU_SCALING_FACTOR * (xi + cube);
                #if !defined(PRECISE_GELU_TANH) && __CUDA_ARCH__ >= 750
                asm ("tanh.approx.f32 %0,%1;" : "=f"(tanh_in_out) : "f"(tanh_in_out));
                #else
                tanh_in_out = tanhf(tanh_in_out);
                #endif

                // the following uses FMUL+FMA instead of FMUL+FADD+FMUL for "0.5f * x * (1.0f + tanh_out)"
                float half_xi = 0.5f * xi;
                packed_out[k] = (floatX)(half_xi * tanh_in_out + half_xi);
            }
            // store instead of storecs (without cache streaming) in case it is useful for the
            // data to be in the cache for the next operation after this GeLU
            store128(out + idx, packed_out);

            int to_next = c_to_next[chunk_id];
            idx += to_next * blockDim.x * x128::size;
            chunk_id = (chunk_id + to_next) & 1023;
            j += to_next;
        }
    }

    #pragma unroll 8
    while(j < iter_per_block) {
        /*
        unsigned int clock, clock2;
        asm volatile("mov.u32 %0, %%clock;" : "=r"(clock));
        x128 packed_inp = load128cs(inp + idx); // load and do not keep in cache
        if (packed_inp[0] == (floatX)999999.99999f) {
            return; // HACK for timing
        }
        asm volatile("mov.u32 %0, %%clock;" : "=r"(clock2));
        unsigned int elapsed_clocks = clock2 - clock;
        if (threadIdx.x == 0) {
            printf("%d: %d (is_far: %d / sm_side_near: %d)\n", idx, elapsed_clocks, is_far, sm_side_near);
        }
        */


        x128 packed_out;
        x128 packed_inp = load128cs(inp + idx); // load and do not keep in cache
        #pragma unroll
        for(int k = 0; k < packed_inp.size; ++k) {
            float xi = (float)packed_inp[k];
            float cube = 0.044715f * xi * xi * xi;

            float tanh_in_out = GELU_SCALING_FACTOR * (xi + cube);
            #if !defined(PRECISE_GELU_TANH) && __CUDA_ARCH__ >= 750
            asm ("tanh.approx.f32 %0,%1;" : "=f"(tanh_in_out) : "f"(tanh_in_out));
            #else
            tanh_in_out = tanhf(tanh_in_out);
            #endif

            // the following uses FMUL+FMA instead of FMUL+FADD+FMUL for "0.5f * x * (1.0f + tanh_out)"
            float half_xi = 0.5f * xi;
            packed_out[k] = (floatX)(half_xi * tanh_in_out + half_xi);
        }
        // store instead of storecs (without cache streaming) in case it is useful for the
        // data to be in the cache for the next operation after this GeLU
        store128(out + idx, packed_out);

        int to_next = c_to_next[chunk_id];
        idx += to_next * blockDim.x * x128::size;
        chunk_id = (chunk_id + to_next) & 1023;
        j += to_next;

        //idx += blockDim.x * x128::size; // = 1 chunk of 4KiB
        //chunk_id = (chunk_id + 1) & 1023;
    }
}



constexpr int per_clean_iter = 8;
constexpr int parallel_blocks = 3;

// Optimised with option to use optimised HW TANH instruction by default
__global__ __launch_bounds__(256, parallel_blocks)
void gelu_forward_kernel4c(floatX* __restrict__ out, const floatX* __restrict__ inp,
                          int iter_per_block, int elements_per_block, int num_near, int num_far,
                          int start_chunk_extra, int iter_per_block_extra, int elements_per_block_extra,
                          unsigned int* __restrict__ num_blocks_active, int N) {
    // get SM id
    unsigned int smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    unsigned int blocks_per_sm = parallel_blocks; //(2048 / blockDim.x);

    __shared__ unsigned int shared_block_in_sm;
    if (threadIdx.x == 0) {
        shared_block_in_sm  = atomicInc(num_blocks_active + smid, blocks_per_sm-1);
    }

    unsigned int sm_side_raw = c_sm_side[smid];
    unsigned int sm_side_near = sm_side_raw & 1;
    unsigned int sm_side_id = sm_side_raw >> 1;
    unsigned int min_per_side = min(num_near, num_far);

    __syncthreads();
    unsigned int block_in_sm = shared_block_in_sm;
    int effective_block_id = (sm_side_id * blocks_per_sm) + block_in_sm;
    //assert(block_in_sm < parallel_blocks);

    if (sm_side_id >= min_per_side) {
        // todo - stragglers do X% + dynamic???
        // use nanosleep to make sure all the other thread blocks have launched
        // and this won't result in this SM getting more than its expected number of thread blocks
        __nanosleep(20000);
        return;

        effective_block_id -= min_per_side * blocks_per_sm;
        size_t idx = start_chunk_extra * (4096 / sizeof(floatX));
        idx += (effective_block_id * elements_per_block_extra) + (threadIdx.x * x128::size);

        #pragma unroll 4
        for(int j = 0; j < iter_per_block_extra; j++) {
            x128 packed_out;
            x128 packed_inp = load128cs(inp + idx); // load and do not keep in cache
            #pragma unroll
            for(int k = 0; k < packed_inp.size; ++k) {
                float xi = (float)packed_inp[k];
                float cube = 0.044715f * xi * xi * xi;

                float tanh_in_out = GELU_SCALING_FACTOR * (xi + cube);
                #if !defined(PRECISE_GELU_TANH) && __CUDA_ARCH__ >= 750
                asm ("tanh.approx.f32 %0,%1;" : "=f"(tanh_in_out) : "f"(tanh_in_out));
                #else
                tanh_in_out = tanhf(tanh_in_out);
                #endif

                // the following uses FMUL+FMA instead of FMUL+FADD+FMUL for "0.5f * x * (1.0f + tanh_out)"
                float half_xi = 0.5f * xi;
                packed_out[k] = (floatX)(half_xi * tanh_in_out + half_xi);
            }
            // store instead of storecs (without cache streaming) in case it is useful for the
            // data to be in the cache for the next operation after this GeLU
            store128(out + idx, packed_out);
            idx += blockDim.x * x128::size;
        }
        return;
    }

    size_t idx = (effective_block_id * elements_per_block) + (threadIdx.x * x128::size);

    //if (smid != 43 || block_in_sm != 0 || threadIdx.x >= 32) {
    //    return;
    //}

    int j = 0;
    size_t addr = (size_t)(inp + idx);
    int chunk_id = (addr >> 12) & 1023;
    unsigned char is_far = c_is_far[chunk_id];

    //while (is_far != (sm_side_near ^ (effective_block_id & 1))) {
    //while (is_far != sm_side_near) {
    while (is_far == sm_side_near) {
        idx += blockDim.x * x128::size;
        chunk_id = (chunk_id + 1) & 1023;
        is_far = c_is_far[chunk_id];
        j++;
    }

    int clean_iters = (iter_per_block / (2 * per_clean_iter)) - 1;

    #pragma unroll 1
    for (int x = 0; x < clean_iters; x++) {
        x128 packed_inps[per_clean_iter];
        //const floatX* in_addrs[per_clean_iter];
        floatX* out_addrs[per_clean_iter];

        #pragma unroll
        for (int y = 0; y < per_clean_iter; y++) {
            //in_addrs[y] = inp + idx;
            packed_inps[y] = load128cs(inp + idx); // load and do not keep in cache
            out_addrs[y] = out + idx;

            int to_next = c_to_next[chunk_id];
            idx += to_next * blockDim.x * x128::size;
            chunk_id = (chunk_id + to_next) & 1023;
            j += to_next;
        }

        __syncthreads();
        __threadfence_block();

        #pragma unroll
        for (int y = 0; y < per_clean_iter; y++) {
            //packed_inps[y] = load128cs(in_addrs[y]); // load and do not keep in cache
        }

        #pragma unroll
        for (int y = 0; y < per_clean_iter; y++) {
            x128 packed_out;
            x128 packed_inp = packed_inps[y];
            #pragma unroll
            for(int k = 0; k < packed_inp.size; ++k) {
                float xi = (float)packed_inp[k];
                float cube = 0.044715f * xi * xi * xi;

                float tanh_in_out = GELU_SCALING_FACTOR * (xi + cube);
                #if !defined(PRECISE_GELU_TANH) && __CUDA_ARCH__ >= 750
                asm ("tanh.approx.f32 %0,%1;" : "=f"(tanh_in_out) : "f"(tanh_in_out));
                #else
                tanh_in_out = tanhf(tanh_in_out);
                #endif

                // the following uses FMUL+FMA instead of FMUL+FADD+FMUL for "0.5f * x * (1.0f + tanh_out)"
                float half_xi = 0.5f * xi;
                packed_out[k] = (floatX)(half_xi * tanh_in_out + half_xi);
            }
            // store instead of storecs (without cache streaming) in case it is useful for the
            // data to be in the cache for the next operation after this GeLU
            store128(out_addrs[y], packed_out);
        }
    }

    #pragma unroll 4
    while(j < iter_per_block) {
        /*
        unsigned int clock, clock2;
        asm volatile("mov.u32 %0, %%clock;" : "=r"(clock));
        x128 packed_inp = load128cs(inp + idx); // load and do not keep in cache
        if (packed_inp[0] == (floatX)999999.99999f) {
            return; // HACK for timing
        }
        asm volatile("mov.u32 %0, %%clock;" : "=r"(clock2));
        unsigned int elapsed_clocks = clock2 - clock;
        if (threadIdx.x == 0) {
            printf("%d: %d (is_far: %d / sm_side_near: %d)\n", idx, elapsed_clocks, is_far, sm_side_near);
        }
        */


        x128 packed_out;
        x128 packed_inp = load128cs(inp + idx); // load and do not keep in cache
        #pragma unroll
        for(int k = 0; k < packed_inp.size; ++k) {
            float xi = (float)packed_inp[k];
            float cube = 0.044715f * xi * xi * xi;

            float tanh_in_out = GELU_SCALING_FACTOR * (xi + cube);
            #if !defined(PRECISE_GELU_TANH) && __CUDA_ARCH__ >= 750
            asm ("tanh.approx.f32 %0,%1;" : "=f"(tanh_in_out) : "f"(tanh_in_out));
            #else
            tanh_in_out = tanhf(tanh_in_out);
            #endif

            // the following uses FMUL+FMA instead of FMUL+FADD+FMUL for "0.5f * x * (1.0f + tanh_out)"
            float half_xi = 0.5f * xi;
            packed_out[k] = (floatX)(half_xi * tanh_in_out + half_xi);
        }
        // store instead of storecs (without cache streaming) in case it is useful for the
        // data to be in the cache for the next operation after this GeLU
        store128(out + idx, packed_out);

        int to_next = c_to_next[chunk_id];
        idx += to_next * blockDim.x * x128::size;
        chunk_id = (chunk_id + to_next) & 1023;
        j += to_next;

        //idx += blockDim.x * x128::size; // = 1 chunk of 4KiB
        //chunk_id = (chunk_id + 1) & 1023;
    }
}

// Optimised with option to use optimised HW TANH instruction by default
__global__ __launch_bounds__(256, parallel_blocks)
void gelu_forward_kernel4(floatX* __restrict__ out, const floatX* __restrict__ inp,
                          int iter_per_block, int elements_per_block, int num_near, int num_far,
                          int start_chunk_extra, int iter_per_block_extra, int elements_per_block_extra,
                          unsigned int* __restrict__ num_blocks_active, int N) {
    // get SM id
    unsigned int smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    unsigned int blocks_per_sm = parallel_blocks; //(2048 / blockDim.x);

    __shared__ unsigned int shared_block_in_sm;
    if (threadIdx.x == 0) {
        shared_block_in_sm  = atomicInc(num_blocks_active + smid, blocks_per_sm-1);
    }

    unsigned int sm_side_raw = c_sm_side[smid];
    unsigned int sm_side_near = sm_side_raw & 1;
    unsigned int sm_side_id = sm_side_raw >> 1;
    unsigned int min_per_side = min(num_near, num_far);

    __syncthreads();
    unsigned int block_in_sm = shared_block_in_sm;
    int effective_block_id = (sm_side_id * blocks_per_sm) + block_in_sm;
    //assert(block_in_sm < parallel_blocks);

    if (sm_side_id >= min_per_side) {
        // todo - stragglers do X% + dynamic???
        // use nanosleep to make sure all the other thread blocks have launched
        // and this won't result in this SM getting more than its expected number of thread blocks
        __nanosleep(20000);
        return;

        effective_block_id -= min_per_side * blocks_per_sm;
        size_t idx = start_chunk_extra * (4096 / sizeof(floatX));
        idx += (effective_block_id * elements_per_block_extra) + (threadIdx.x * x128::size);

        #pragma unroll 4
        for(int j = 0; j < iter_per_block_extra; j++) {
            x128 packed_out;
            x128 packed_inp = load128cs(inp + idx); // load and do not keep in cache
            #pragma unroll
            for(int k = 0; k < packed_inp.size; ++k) {
                float xi = (float)packed_inp[k];
                float cube = 0.044715f * xi * xi * xi;

                float tanh_in_out = GELU_SCALING_FACTOR * (xi + cube);
                #if !defined(PRECISE_GELU_TANH) && __CUDA_ARCH__ >= 750
                asm ("tanh.approx.f32 %0,%1;" : "=f"(tanh_in_out) : "f"(tanh_in_out));
                #else
                tanh_in_out = tanhf(tanh_in_out);
                #endif

                // the following uses FMUL+FMA instead of FMUL+FADD+FMUL for "0.5f * x * (1.0f + tanh_out)"
                float half_xi = 0.5f * xi;
                packed_out[k] = (floatX)(half_xi * tanh_in_out + half_xi);
            }
            // store instead of storecs (without cache streaming) in case it is useful for the
            // data to be in the cache for the next operation after this GeLU
            store128(out + idx, packed_out);
            idx += blockDim.x * x128::size;
        }
        return;
    }

    size_t idx = (effective_block_id * elements_per_block) + (threadIdx.x * x128::size);

    //if (smid != 43 || block_in_sm != 0 || threadIdx.x >= 32) {
    //    return;
    //}

    int j = 0;
    size_t addr = (size_t)(inp + idx);
    int chunk_id = (addr >> 12) & 1023;
    unsigned char is_far = c_is_far[chunk_id];

    //while (is_far != (sm_side_near ^ (effective_block_id & 1))) {
    //while (is_far != sm_side_near) {
    while (is_far == sm_side_near) {
        idx += blockDim.x * x128::size;
        chunk_id = (chunk_id + 1) & 1023;
        is_far = c_is_far[chunk_id];
        j++;
    }

    int clean_iters = (iter_per_block / (2 * per_clean_iter)) - 1;

    #pragma unroll 1
    for (int x = 0; x < clean_iters; x++) {
        x128 packed_inps[per_clean_iter];
        //const floatX* in_addrs[per_clean_iter];
        floatX* out_addrs[per_clean_iter];

        #pragma unroll
        for (int y = 0; y < per_clean_iter; y++) {
            //in_addrs[y] = inp + idx;
            packed_inps[y] = load128cs(inp + idx); // load and do not keep in cache
            out_addrs[y] = out + idx;

            int to_next = c_to_next[chunk_id];
            idx += to_next * blockDim.x * x128::size;
            chunk_id = (chunk_id + to_next) & 1023;
            j += to_next;
        }

        __syncthreads();
        __threadfence_block();

        #pragma unroll
        for (int y = 0; y < per_clean_iter; y++) {
            //packed_inps[y] = load128cs(in_addrs[y]); // load and do not keep in cache
        }

        #pragma unroll
        for (int y = 0; y < per_clean_iter; y++) {
            x128 packed_out;
            x128 packed_inp = packed_inps[y];
            #pragma unroll
            for(int k = 0; k < packed_inp.size; ++k) {
                float xi = (float)packed_inp[k];
                float cube = 0.044715f * xi * xi * xi;

                float tanh_in_out = GELU_SCALING_FACTOR * (xi + cube);
                #if !defined(PRECISE_GELU_TANH) && __CUDA_ARCH__ >= 750
                asm ("tanh.approx.f32 %0,%1;" : "=f"(tanh_in_out) : "f"(tanh_in_out));
                #else
                tanh_in_out = tanhf(tanh_in_out);
                #endif

                // the following uses FMUL+FMA instead of FMUL+FADD+FMUL for "0.5f * x * (1.0f + tanh_out)"
                float half_xi = 0.5f * xi;
                packed_out[k] = (floatX)(half_xi * tanh_in_out + half_xi);
            }
            // store instead of storecs (without cache streaming) in case it is useful for the
            // data to be in the cache for the next operation after this GeLU
            store128(out_addrs[y], packed_out);
        }
    }

    #pragma unroll 4
    while(j < iter_per_block) {
        /*
        unsigned int clock, clock2;
        asm volatile("mov.u32 %0, %%clock;" : "=r"(clock));
        x128 packed_inp = load128cs(inp + idx); // load and do not keep in cache
        if (packed_inp[0] == (floatX)999999.99999f) {
            return; // HACK for timing
        }
        asm volatile("mov.u32 %0, %%clock;" : "=r"(clock2));
        unsigned int elapsed_clocks = clock2 - clock;
        if (threadIdx.x == 0) {
            printf("%d: %d (is_far: %d / sm_side_near: %d)\n", idx, elapsed_clocks, is_far, sm_side_near);
        }
        */


        x128 packed_out;
        x128 packed_inp = load128cs(inp + idx); // load and do not keep in cache
        #pragma unroll
        for(int k = 0; k < packed_inp.size; ++k) {
            float xi = (float)packed_inp[k];
            float cube = 0.044715f * xi * xi * xi;

            float tanh_in_out = GELU_SCALING_FACTOR * (xi + cube);
            #if !defined(PRECISE_GELU_TANH) && __CUDA_ARCH__ >= 750
            asm ("tanh.approx.f32 %0,%1;" : "=f"(tanh_in_out) : "f"(tanh_in_out));
            #else
            tanh_in_out = tanhf(tanh_in_out);
            #endif

            // the following uses FMUL+FMA instead of FMUL+FADD+FMUL for "0.5f * x * (1.0f + tanh_out)"
            float half_xi = 0.5f * xi;
            packed_out[k] = (floatX)(half_xi * tanh_in_out + half_xi);
        }
        // store instead of storecs (without cache streaming) in case it is useful for the
        // data to be in the cache for the next operation after this GeLU
        store128(out + idx, packed_out);

        int to_next = c_to_next[chunk_id];
        idx += to_next * blockDim.x * x128::size;
        chunk_id = (chunk_id + to_next) & 1023;
        j += to_next;

        //idx += blockDim.x * x128::size; // = 1 chunk of 4KiB
        //chunk_id = (chunk_id + 1) & 1023;
    }
}

__global__ __launch_bounds__(256, parallel_blocks)
void gelu_forward_kernel5(floatX* out, const floatX* inp,
                          int clean_iter_per_block, int plausible_iter_per_block,
                          int iter_per_block, int elements_per_block, int num_near, int num_far,
                          int start_chunk_extra, int iter_per_block_extra, int elements_per_block_extra,
                          unsigned int* num_blocks_active, int N) {
    // get SM id
    unsigned int smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    unsigned int blocks_per_sm = parallel_blocks; //(2048 / blockDim.x);

    int elements_per_4KiB = 4*1024/sizeof(floatX);
    int elements_per_2MiB = 2*1024*1024/sizeof(floatX);

    __shared__ unsigned int shared_block_in_sm;
    if (threadIdx.x == 0) {
        shared_block_in_sm  = atomicInc(num_blocks_active + smid, blocks_per_sm-1);
    }

    unsigned int sm_side_raw = c_sm_side[smid];
    unsigned int sm_side_is_near = sm_side_raw & 1;
    unsigned int sm_side_id = sm_side_raw >> 1;
    unsigned int min_per_side = min(num_near, num_far);

    __syncthreads();
    unsigned int block_in_sm = shared_block_in_sm;
    int effective_block_id = (sm_side_id * blocks_per_sm) + block_in_sm;
    //assert(block_in_sm < parallel_blocks);

    if (sm_side_id >= min_per_side) {
        // todo - stragglers do X% + dynamic???
        __nanosleep(20000);
        return;
    }

    int blocks_per_side = min_per_side * blocks_per_sm;
    int stride = elements_per_4KiB * blocks_per_side;
    unsigned int idx = effective_block_id * elements_per_4KiB + threadIdx.x * x128::size;

    floatX* original_out = out;

    int chunk_id = ((size_t)(inp + idx) >> 12) & 1023;
    idx += (chunk_id >= 512) ? elements_per_2MiB : 0;
    chunk_id &= 511;

    #pragma unroll 1
    for (int x = 0; x < clean_iter_per_block; x += per_clean_iter) {
        x128 packed_inps[per_clean_iter];
        //const floatX* in_addrs[per_clean_iter];
        floatX* out_addrs[per_clean_iter];

        #pragma unroll
        for (int y = 0; y < per_clean_iter; y++) {
            unsigned int idx_plus_2MiB = idx + elements_per_2MiB;
            unsigned int idx2 = (c_is_far[chunk_id] == sm_side_is_near) ? idx_plus_2MiB : idx;
            packed_inps[y] = load128cs(inp + idx2);
            out_addrs[y] = out + idx2;

            chunk_id = (chunk_id + blocks_per_side);
            idx += stride + ((chunk_id >= 512) ? elements_per_2MiB : 0);
            chunk_id &= 511;
        }
        inp += idx;
        out += idx;
        idx = 0;

        __syncthreads();
        __threadfence_block();

        #pragma unroll
        for (int y = 0; y < per_clean_iter; y++) {
            x128 packed_out;
            x128 packed_inp = packed_inps[y];
            #pragma unroll
            for(int k = 0; k < packed_inp.size; ++k) {
                float xi = (float)packed_inp[k];
                float cube = 0.044715f * xi * xi * xi;

                float tanh_in_out = GELU_SCALING_FACTOR * (xi + cube);
                #if !defined(PRECISE_GELU_TANH) && __CUDA_ARCH__ >= 750
                asm ("tanh.approx.f32 %0,%1;" : "=f"(tanh_in_out) : "f"(tanh_in_out));
                #else
                tanh_in_out = tanhf(tanh_in_out);
                #endif

                // the following uses FMUL+FMA instead of FMUL+FADD+FMUL for "0.5f * x * (1.0f + tanh_out)"
                float half_xi = 0.5f * xi;
                packed_out[k] = (floatX)(half_xi * tanh_in_out + half_xi);
            }
            // store instead of storecs (without cache streaming) in case it is useful for the
            // data to be in the cache for the next operation after this GeLU
            store128(out_addrs[y], packed_out);
        }
    }

    size_t real_idx = (size_t)(out - original_out);

    int j = 0;
    for(int j = 0; j < plausible_iter_per_block; j++) {
        unsigned int offset = (c_is_far[chunk_id] == sm_side_is_near) ? elements_per_2MiB : 0;

        if (real_idx + offset < N) {
            x128 packed_out;
            x128 packed_inp = load128cs(inp + offset + idx);
            #pragma unroll
            for(int k = 0; k < packed_inp.size; ++k) {
                float xi = (float)packed_inp[k];
                float cube = 0.044715f * xi * xi * xi;

                float tanh_in_out = GELU_SCALING_FACTOR * (xi + cube);
                #if !defined(PRECISE_GELU_TANH) && __CUDA_ARCH__ >= 750
                asm ("tanh.approx.f32 %0,%1;" : "=f"(tanh_in_out) : "f"(tanh_in_out));
                #else
                tanh_in_out = tanhf(tanh_in_out);
                #endif

                // the following uses FMUL+FMA instead of FMUL+FADD+FMUL for "0.5f * x * (1.0f + tanh_out)"
                float half_xi = 0.5f * xi;
                packed_out[k] = (floatX)(half_xi * tanh_in_out + half_xi);
            }
            // store instead of storecs (without cache streaming) in case it is useful for the
            // data to be in the cache for the next operation after this GeLU
            store128(out + offset + idx, packed_out);
        }

        chunk_id += blocks_per_side;
        idx += stride + (chunk_id >= 512 ? elements_per_2MiB : 0);
        chunk_id &= 511;
    }
}




// ----------------------------------------------------------------------------
// kernel launcher

void gelu_forward1(floatX* out, const floatX* inp, int N, const int block_size) {
    const int grid_size = ceil_div(N, block_size);
    gelu_forward_kernel1<<<grid_size, block_size>>>(out, inp, N);
    cudaCheck(cudaGetLastError());
}

void gelu_forward2(floatX* out, const floatX* inp, int N, const int block_size) {
    const int grid_size = ceil_div(N, block_size * x128::size);
    gelu_forward_kernel2<<<grid_size, block_size>>>(out, inp, N);
    cudaCheck(cudaGetLastError());
}

void gelu_forward3(floatX* out, const floatX* inp, int N, const int block_size) {
    const int grid_size = ceil_div(N, block_size * x128::size);
    gelu_forward_kernel3<<<grid_size, block_size>>>(out, inp, N);
    cudaCheck(cudaGetLastError());
}

constexpr float use_wasted_multiplier = 0.0f;

void gelu_forward4(floatX* out, const floatX* inp, int N, int block_size) {
    /*int blocks_per_sm = num_threads_per_sm / block_size;
    int grid_size = num_sms * blocks_per_sm;
    int iter_per_block = ceil_div(N, grid_size * block_size * x128::size);
    int elements_per_block = iter_per_block * block_size * x128::size;*/

    block_size = 256; // hack
    //int blocks_per_sm = (num_threads_per_sm / block_size);
    int blocks_per_sm = parallel_blocks; // hack

    int elements_per_4kib_chunk = 4096 / sizeof(floatX);
    int num_4kib_chunks = ceil_div(N, elements_per_4kib_chunk);
    int min_per_side = min(num_near, num_far);

    int wasted_SMs = num_sms - (2 * min_per_side);
    float percentage_wasted = (float)wasted_SMs / (float)num_sms;
    float adjusted_used = 1.0f - (use_wasted_multiplier * percentage_wasted);

    int num_4kib_chunks_main = (int)ceilf((float)num_4kib_chunks * adjusted_used);
    int num_4kib_chunks_extra = num_4kib_chunks - num_4kib_chunks_main;

    int effective_num_blocks = min_per_side * blocks_per_sm;
    int iter_per_block = ceil_div(num_4kib_chunks_main, effective_num_blocks);
    int elements_per_block = elements_per_4kib_chunk * iter_per_block;

    int effective_num_blocks_extra = wasted_SMs * blocks_per_sm;
    int iter_per_block_extra = ceil_div(num_4kib_chunks_extra, effective_num_blocks_extra);
    int elements_per_block_extra = elements_per_4kib_chunk * iter_per_block_extra;

    /*
    printf("\n==========\n");
    printf("N: %d\n", N);
    printf("num_4kib_chunks: %d\n", num_4kib_chunks);
    printf("num_4kib_chunks_main: %d\n", num_4kib_chunks_main);
    printf("num_4kib_chunks_extra: %d\n", num_4kib_chunks_extra);
    printf("iter_per_block: %d\n", iter_per_block);
    printf("elements_per_block: %d\n", elements_per_block);
    printf("iter_per_block_extra: %d\n", iter_per_block_extra);
    printf("elements_per_block_extra: %d\n", elements_per_block_extra);
    printf("effective_num_blocks: %d\n", effective_num_blocks);
    printf("effective_num_blocks_extra: %d\n", effective_num_blocks_extra);
    printf("SMs: %d, min_per_side: %d, wasted_SMs: %d\n", num_sms, min_per_side, wasted_SMs);
    printf("percentage_wasted: %f\n", percentage_wasted);
    printf("adjusted_used: %f\n", adjusted_used);
    printf("==========\n\n");
    */

    int inner_loop_iter = elements_per_4kib_chunk / (block_size * x128::size);
    assert(inner_loop_iter == 1);

    //cudaMemset(num_blocks_active, 0, num_sms * 64 * sizeof(unsigned int));

    int grid_size = num_sms * blocks_per_sm;
    gelu_forward_kernel4<<<grid_size, block_size>>>(out, inp, iter_per_block, elements_per_block, num_near, num_far,
                                                    num_4kib_chunks_main, iter_per_block_extra, elements_per_block_extra,
                                                    num_blocks_active, N);
    cudaCheck(cudaGetLastError());
}

void gelu_forward5(floatX* out, const floatX* inp, int N, int block_size) {
    /*int blocks_per_sm = num_threads_per_sm / block_size;
    int grid_size = num_sms * blocks_per_sm;
    int iter_per_block = ceil_div(N, grid_size * block_size * x128::size);
    int elements_per_block = iter_per_block * block_size * x128::size;*/

    block_size = 256; // hack
    //int blocks_per_sm = (num_threads_per_sm / block_size);
    int blocks_per_sm = parallel_blocks; // hack

    int elements_per_4kib_chunk = 4096 / sizeof(floatX);
    int num_4kib_chunks = ceil_div(N, elements_per_4kib_chunk);
    int min_per_side = min(num_near, num_far);

    int wasted_SMs = num_sms - (2 * min_per_side);
    float percentage_wasted = (float)wasted_SMs / (float)num_sms;
    float adjusted_used = 1.0f - (use_wasted_multiplier * percentage_wasted);
    int num_4kib_chunks_main = (int)ceilf((float)num_4kib_chunks * adjusted_used);






    int effective_num_blocks = 2 * min_per_side * blocks_per_sm;
    int iter_per_block = ceil_div(num_4kib_chunks_main, effective_num_blocks);
    int elements_per_block = elements_per_4kib_chunk * iter_per_block;

    int safe_4kib_chunks_main = max(0, num_4kib_chunks_main - 512);
    int clean_iter_per_block = ((safe_4kib_chunks_main / effective_num_blocks) / per_clean_iter) * per_clean_iter;

    int plausible_4kib_chunks_main = ceil_div(num_4kib_chunks_main, 1024) * 1024; // todo - optimise
    int plausible_iter_per_block = (plausible_4kib_chunks_main / effective_num_blocks) - clean_iter_per_block + 1;






    int num_4kib_chunks_extra = num_4kib_chunks - num_4kib_chunks_main;
    int effective_num_blocks_extra = wasted_SMs * blocks_per_sm;
    int iter_per_block_extra = ceil_div(num_4kib_chunks_extra, effective_num_blocks_extra);
    int elements_per_block_extra = elements_per_4kib_chunk * iter_per_block_extra;

    /*
    printf("\n==========\n");
    printf("N: %d\n", N);
    printf("num_4kib_chunks: %d\n", num_4kib_chunks);
    printf("num_4kib_chunks_main: %d\n", num_4kib_chunks_main);
    printf("num_4kib_chunks_extra: %d\n", num_4kib_chunks_extra);
    printf("iter_per_block: %d\n", iter_per_block);
    printf("elements_per_block: %d\n", elements_per_block);
    printf("iter_per_block_extra: %d\n", iter_per_block_extra);
    printf("elements_per_block_extra: %d\n", elements_per_block_extra);
    printf("effective_num_blocks: %d\n", effective_num_blocks);
    printf("effective_num_blocks_extra: %d\n", effective_num_blocks_extra);
    printf("SMs: %d, min_per_side: %d, wasted_SMs: %d\n", num_sms, min_per_side, wasted_SMs);
    printf("percentage_wasted: %f\n", percentage_wasted);
    printf("adjusted_used: %f\n", adjusted_used);
    printf("==========\n\n");
    */

    int inner_loop_iter = elements_per_4kib_chunk / (block_size * x128::size);
    assert(inner_loop_iter == 1);

    //cudaMemset(num_blocks_active, 0, num_sms * 64 * sizeof(unsigned int));

    cudaCheck(cudaDeviceSynchronize());
    int grid_size = num_sms * blocks_per_sm;

    //printf("grid_size: %d, block_size: %d, clean_iter_per_block: %d, plausible_iter_per_block: %d, iter_per_block: %d, elements_per_block: %d, num_near: %d, num_far: %d, num_4kib_chunks_main: %d, iter_per_block_extra: %d, elements_per_block_extra: %d, N: %d\n",
    //       grid_size, block_size, clean_iter_per_block, plausible_iter_per_block, iter_per_block, elements_per_block, num_near, num_far, num_4kib_chunks_main, iter_per_block_extra, elements_per_block_extra, N);

    gelu_forward_kernel5<<<grid_size, block_size>>>(out, inp, clean_iter_per_block, plausible_iter_per_block,
                                                    iter_per_block, elements_per_block, num_near, num_far,
                                                    num_4kib_chunks_main, iter_per_block_extra, elements_per_block_extra,
                                                    num_blocks_active, N);
    cudaCheck(cudaDeviceSynchronize());
    cudaCheck(cudaGetLastError());
}

void latency_check(floatX* out, unsigned char* block_is_far, int N) {
    const int grid_size = 1;
    latency_kernel<<<grid_size, 32>>>(out, block_is_far, N);
    cudaCheck(cudaGetLastError());
}

void sm_side_check(floatX* out, unsigned char* block_is_far, unsigned char* sm_side, int num_sms) {
    sm_kernel<<<num_sms, 256>>>(out, block_is_far, sm_side);
    cudaCheck(cudaGetLastError());
}

// kernel version dispatch
void gelu_forward(int kernel_num,
                  floatX* out,
                  const floatX* inp,
                  int B, int T, int C,
                  int block_size) {
    switch (kernel_num) {
        case 1:
            gelu_forward1(out, inp, B * T * C, block_size);
            break;
        case 2:
            gelu_forward2(out, inp, B * T * C, block_size);
            break;
        case 3:
            gelu_forward3(out, inp, B * T * C, block_size);
            break;
        case 4:
            gelu_forward4(out, inp, B * T * C, block_size);
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

    // cudaMalloc gpu_scratch_l2_clear
    cudaCheck(cudaMalloc(&gpu_scratch_l2_clear, l2_clear_size));

    // create host memory of random numbers
    float* out = (float*)malloc(B * T * C * sizeof(float));
    float* inp = make_random_float(B * T * C);

    // read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);

    // first check the correctness of the kernel
    gelu_forward_cpu(out, inp, B * T * C);

    // move to GPU
    floatX* d_out;
    floatX* d_inp;
    allocateCompressible((void**)&d_out, B * T * C * sizeof(floatX) * 2, enable_compression);
    allocateCompressible((void**)&d_inp, B * T * C * sizeof(floatX) * 2, enable_compression);
    //cudaCheck(cudaMalloc(&d_out, B * T * C * sizeof(floatX)));
    //cudaCheck(cudaMalloc(&d_inp, B * T * C * sizeof(floatX)));
    cudaCheck(memcpy_convert(d_inp, inp, B * T * C));

    cudaCheck(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0));
    cudaCheck(cudaDeviceGetAttribute(&num_threads_per_sm, cudaDevAttrMaxThreadsPerMultiProcessor, 0));

    assert(!cuMemHostAlloc((void**)&block_is_far, 2048 * sizeof(unsigned char), 0));
    cudaCheck(cudaMallocHost((void**)&sm_side, num_sms * sizeof(unsigned char)));

    cudaCheck(cudaMallocHost((void**)&num_blocks_active, num_sms * 64 * sizeof(unsigned int)));
    cudaMemset(num_blocks_active, 0, num_sms * 64 * sizeof(unsigned int));
    cudaMemset(gpu_scratch_l2_clear, 0, l2_clear_size);

    latency_check(d_inp, block_is_far, B * T * C);
    cudaCheck(cudaDeviceSynchronize());
    for (int i = 0; i < 2048; i++) {
        printf("%d", block_is_far[i]);
        if ((i % 512) == 511) {
            printf("\n");
        }
    }

    // Let's figure out which SMs are on the "near" and "far" sides
    // 0) Clear L2 cache
    cudaMemset(gpu_scratch_l2_clear, 0, l2_clear_size);
    // 1) Get number of SMs/multiprocessors
    cudaCheck(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0));
    printf("Number of SMs: %d\n", num_sms);
    // 3) Allocate SM side
    // 4) Launch kernel to get SM IDs
    sm_side_check(d_inp, block_is_far, sm_side, num_sms);
    cudaCheck(cudaDeviceSynchronize());

    unsigned char to_next[1024] = {0};
    for (int i = 0; i < 1024; i++) {
        unsigned char is_far = block_is_far[i];
        for (int j = i+1; j < 2048; j++) {
            if (block_is_far[j] == is_far) {
                to_next[i] = j - i;
                break;
            }
        }
    }

    for (int i = 0; i < num_sms; i++) {
        if (sm_side[i] == 0) {
            sm_side[i] |= num_near << 1;
            num_near++;
        } else {
            sm_side[i] |= num_far << 1;
            num_far++;
        }
        //printf("SM %d: %d/%d\n", i, sm_side[i] & 1, sm_side[i] >> 1);
    }

    // Print all of is_far
    /*for (int i = 0; i < 1024; i++) {
        printf("is_far[%d]: %d\n", i, block_is_far[i]);
    }*/

    cudaMemcpyToSymbol(c_sm_side, sm_side, num_sms * sizeof(unsigned char));
    cudaMemcpyToSymbol(c_is_far, block_is_far, 1024 * sizeof(unsigned char));
    cudaMemcpyToSymbol(c_to_next, to_next, 1024 * sizeof(unsigned char));
    cudaCheck(cudaDeviceSynchronize());

    // time the kernel at different block sizes
    int block_sizes[] = {128, 256};
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        // memset d_out
        cudaMemset(d_out, 0, B * T * C * sizeof(floatX));

        int block_size = block_sizes[j];
        printf("Checking block size %d.\n", block_size);
        gelu_forward(kernel_num, d_out, d_inp, B, T, C, block_size);
#if !defined(ENABLE_BF16) && !defined(ENABLE_FP16)
        float tol = 1e-5f;
#else
        float tol = 1e-3f;
#endif
        validate_result(d_out, out, "out", B * T * C, tol);
    }

    printf("All results match. Starting benchmarks.\n\n");

    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];

        int repeat_times = 1000;

        float elapsed_time = benchmark_kernel(repeat_times, gelu_forward,
                                              kernel_num, d_out, d_inp,
                                              B, T, C, block_size);

        // napkin math: estimate the memory bandwidth achieved
        // for each (B,T,C) output element, we do 1 read and 1 write, 4 bytes each
        // and e.g. A100 40GB PCIe is advertised at 1,555GB/s
        long memory_ops = B * T * C * 2 * (int)sizeof(floatX);
        float memory_bandwidth = memory_ops / elapsed_time / 1e6;

        printf("block_size %4d | time %.4f ms | bandwidth %.2f GB/s\n", block_size, elapsed_time, memory_bandwidth);
    }

    // free memory
    free(out);
    free(inp);

    freeCompressible(d_out, B * T * C * sizeof(floatX), enable_compression);
    freeCompressible(d_inp, B * T * C * sizeof(floatX), enable_compression);
    //cudaCheck(cudaFree(d_out));
    //cudaCheck(cudaFree(d_inp));
    return 0;
}