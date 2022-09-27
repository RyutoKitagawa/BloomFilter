#include <cub/cub.cuh>
#include <curand.h>
#include <curand_kernel.h>
#include <time.h>

#define BLOCKS 68
#define THREADS 1024

__global__ void randomUnsignedInts(unsigned int* data, int size) {
	// Initialize the random number generator
	curandState state;
	curand_init(clock64(), threadIdx.x, blockIdx.x, &state);

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < size) data[tid] = curand(&state);
}

__global__ void sort(unsigned int* data, int size) {
	typedef cub::BlockRadixSort<unsigned int, 128, 4> BlockRadixSort;

	__shared__ typename BlockRadixSort::TempStorage temp_storage;
	unsigned int thread_keys[4];

	BlockRadixSort(temp_storage).Sort(thread_keys);
}

int main() {
	// Allocate memory on the device
	unsigned int* data, size = 20;
	cudaMalloc(&data, size * sizeof(unsigned int));

	// Launch the kernel
	randomUnsignedInts<<<THREADS, BLOCKS>>>(data, size);
	cudaDeviceSynchronize();

	// Sort the data
	sort<<<BLOCKS, THREADS>>>(data, size);
	cudaDeviceSynchronize();

	// Copy the data back to the host
	unsigned int* hostData = (unsigned int*)malloc(size * sizeof(unsigned int));
	cudaMemcpy(hostData, data, size * sizeof(unsigned int), cudaMemcpyDeviceToHost);

	// Print the data
	for (int i = 0; i < size; i++)
		printf("%u\n", hostData[i]);
}
