#include <stdio.h>
#include <cuda_runtime.h>
#include <getopt.h>
#include "malloc.h"
#include <chrono>
#include <iostream>
#include <cooperative_groups.h>
using namespace cooperative_groups;

typedef struct counters
{
    volatile int malloc_counter;
    volatile int free_counter;
    int thread_count;
}counters_t;

__device__ void atomicAggDec(volatile int *ctr) {
  auto g = coalesced_threads();
  int warp_res;
  if(g.thread_rank() == 0)
    warp_res = atomicSub((int*)ctr, g.size()); // divergence risk
}

__device__ void* linear_cudaMalloc(int size_in_bytes, counters_t *counter, void *g_base_addr)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid == 0)
        cudaMalloc((void**)g_base_addr, size_in_bytes * counter->thread_count);

    atomicAggDec(&counter->malloc_counter);
    while(counter->malloc_counter > 0);
    return (void*)(((*(char**)g_base_addr)) + (size_in_bytes * tid));
}

__device__ void linear_cudaFree(counters_t *counter, void *addr)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    atomicAggDec(&counter->free_counter);
    while(counter->free_counter > 0);

    if(tid == 0)
        cudaFree(addr);
}

__global__ void myKernel(int *output, int n)
{
    int *dev_array;
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    cudaMalloc((void **)&dev_array, n * sizeof(int));

    for (int i = 0; i < n; i++)
    {
        dev_array[i] = (id * i);
    }

    for (int i = 0; i < n; i++)
    {
        output[id] += dev_array[i];
    }
    cudaFree(dev_array);
}

__global__ void linear_malloc_kernel(int *output, int n, counters_t *counter, void *g_base_addr)
{
    int *dev_array;
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    dev_array = (int*)linear_cudaMalloc(n*sizeof(int), counter, g_base_addr);
    printf("%d , %p\n", id, dev_array);
    for (int i = 0; i < n; i++)
    {
        dev_array[i] = (id * i);
    }

    for (int i = 0; i < n; i++)
    {
        output[id] += dev_array[i];
    }
    linear_cudaFree(counter, dev_array);
}

int cpu_val(int max_tid, int malloc_size)
{
    int output = 0;
    for(int tid=0; tid<max_tid; tid++)
    {
        for (int i=0; i<malloc_size; i++)
        {
            output += (tid * i);
        }
    }
    return output;
}

// ./malloc blocks_per_grid, threads_per_block, malloc_size_per_thread_in_int_size
// ./malloc -g -b -n
int main(int argc, char **argv)
{
    int *dev_output;
    int grid_size, block_size, malloc_size;
    int opt;
    while ((opt = getopt(argc, argv, "g:b:n:")) != -1)
    {
        switch (opt)
        {
        case 'g':
            grid_size = atoi(optarg);
            break;
        case 'b':
            block_size = atoi(optarg);
            break;
        case 'n':
            malloc_size = atoi(optarg);
            break;
        case '?':
            printf("Usage: %s [-g grid_size] [-b block_size] [-n malloc_size_per_thread_in_int_size]\n", argv[0]);
            return 0;
        }
    }

    printf("%d, %d, %d\n", grid_size, block_size, malloc_size);
    int data_size = grid_size * block_size * sizeof(int);
    cudaMallocManaged((void **)&dev_output, data_size);

    auto start = std::chrono::high_resolution_clock::now();
    myKernel<<<grid_size, block_size>>>(dev_output, malloc_size);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "cudaMalloc Execution time: " << duration << " milliseconds" << std::endl;

    int output_sum = 0;
    for (int i = 0; i < data_size/sizeof(int); i++)
    {
        output_sum += dev_output[i];
    }
    int cpu_sum = cpu_val(grid_size * block_size, malloc_size);

    printf("gpu_output_sum %d, cpu_sum %d \n", output_sum, cpu_sum);

    if(output_sum != cpu_sum)
        printf("ERROR: CPU and GPU vals don't match!!!\n");
    else
        printf("default malloc-free test PASSED\n");

    counters_t *counter;
    void *g_base_address;
    cudaMallocManaged((void**)&counter, sizeof(counters_t));

    counter->malloc_counter = grid_size * block_size;
    counter->free_counter = grid_size * block_size;

    cudaMallocManaged(&g_base_address, sizeof(void*));

    cudaMemset((void*)dev_output, 0, data_size);

    counter->thread_count = malloc_size * grid_size * block_size;
    start = std::chrono::high_resolution_clock::now();
    linear_malloc_kernel<<<grid_size, block_size>>>(dev_output, malloc_size, counter, g_base_address);
    cudaDeviceSynchronize();

    end = std::chrono::high_resolution_clock::now();

    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "cudaMalloc Execution time: " << duration << " milliseconds" << std::endl;

    output_sum = 0;
    for (int i = 0; i < data_size/sizeof(int); i++)
    {
        output_sum += dev_output[i];
    }
    printf("gpu_output_sum %d, cpu_sum %d \n", output_sum, cpu_sum);

    if(output_sum != cpu_sum)
        printf("ERROR: CPU and GPU vals don't match!!!\n");
    else
        printf("PASSED\n");

    cudaFree(dev_output);

    return 0;
}