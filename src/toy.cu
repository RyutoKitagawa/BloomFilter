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
	if (tid < size) data[tid] = curand(&state) % 100;
}

__global__ void sort(unsigned int* data, int size) {
	typedef cub::BlockRadixSort<unsigned int, THREADS, 4> BlockRadixSort;

	__shared__ typename BlockRadixSort::TempStorage temp_storage;
	unsigned int thread_keys[4];
	for (int i = 0; i < 4; i++) {
		int tid = blockIdx.x * blockDim.x * 4 + threadIdx.x + i;
		if (tid < size) thread_keys[i] = data[tid];
		else thread_keys[i] = 0xFFFFFFFF;
	}

	BlockRadixSort(temp_storage).Sort(thread_keys);

	for (int i = 0; i < 4; i++) {
		int tid = blockIdx.x * blockDim.x * 4 + threadIdx.x + i;
		if (tid < size) data[tid] = thread_keys[i];
	}
}

__global__ void verify(unsigned int* data, int size) {
	for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < size - 1; i += blockDim.x * gridDim.x)
		if (data[i] > data[i + 1])
			printf("Error at %d: %d > %d\n", i, data[i], data[i + 1]);
}

int main() {
	// Allocate memory on the device
	unsigned int* data, size = 1000000;
	cudaMalloc(&data, size * sizeof(unsigned int));

	// Launch the kernel
	randomUnsignedInts<<<THREADS, BLOCKS>>>(data, size);
	cudaDeviceSynchronize();

	// Sort the data
	sort<<<BLOCKS, THREADS>>>(data, size);
	cudaDeviceSynchronize();

	// Verify the data
	verify<<<BLOCKS, THREADS>>>(data, size);
	cudaDeviceSynchronize();

	// Free the memory
	cudaFree(data);
}
