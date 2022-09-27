#ifndef PARALLEL_INVERTABLE_BLOOM_FILTER_CUH
#define PARALLEL_INVERTABLE_BLOOM_FILTER_CUH

#define BLOCKS 68
#define THREADS 1024

#ifdef DEBUG
#define D(command) command;
#define ONE(command) if(threadIdx.x + blockIdx.x == 0) command;
#else
#define ONE(command) ;
#define D(command) ;
#endif

#include <iostream>
#include <vector>
#include <algorithm>
#include <cub/cub.cuh>

#include "device_vector.cuh"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


namespace pibf {
	typedef unsigned int ValueType;

	struct TableType {
		unsigned int count;
		ValueType value;
	};

	__host__ __device__ inline size_t hash(ValueType value, unsigned int seed, size_t size) {
		unsigned int hash = seed;
		hash ^= value + 0x9e3779b9 + (hash << 6) + (hash >> 2);
		return hash % size;
	}

	__host__ __device__ inline unsigned int hash(ValueType value, size_t size) {
		return ((value * 0xDEADBEEF) >> 18) % size;
	}

	template<typename T>
	__device__ inline void printArray(T* array, size_t size) {
		if (threadIdx.x + blockIdx.x == 0)
			for (size_t i = 0; i < size; i++)
				printf("%d ", array[i]);
		if (threadIdx.x + blockIdx.x == 0) printf("\n");
	}

	__global__ void insertKernel(TableType* table, size_t tableSize, ValueType* values, unsigned int valueSize, size_t r) {
		for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < valueSize; i += blockDim.x * gridDim.x) {
			for (int j = 0; j < r; j++) {
				size_t index = hash(values[i], j, tableSize);
				atomicAdd(&table[index].count, 1);
				atomicAdd(&table[index].value, values[i]);
			}
		}
	}

	__global__ void removeKernel(TableType* table, size_t tableSize, ValueType* values, unsigned int valueSize, size_t r) {
		for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < valueSize; i += blockDim.x * gridDim.x) {
			for (int j = 0; j < r; j++) {
				size_t index = hash(values[i], j, tableSize);
				atomicSub(&table[index].count, 1);
				atomicSub(&table[index].value, values[i]);
			}
		}
	}

	// NOTE: Can only peel with less than 278,000 values
	__global__ void peel(TableType* table, size_t tableSize, ValueType* values, size_t valueSize, size_t r) {
		const unsigned int peeledSize = 4;
		unsigned int count = 0;
		ValueType peeled[peeledSize] = {0xFFFFFFFF};

		if (tableSize < BLOCKS * THREADS * peeledSize) return;

		for (int i = threadIdx.x ; i < tableSize; i += blockDim.x * gridDim.x)
			if (table[i].count == 1) peeled[counter++] = table[i].value;

		__syncthreads();
	}

	class ParallelInvertableBloomFilter {
	private:
		size_t n, r, c;
		DeviceVector<TableType> d_table;

	public:
		ParallelInvertableBloomFilter(unsigned int expected_num_elements) {
			this->n = expected_num_elements;
			this->r = (int) std::log2(std::max(std::log2(n), 2.0));
			this->c = 2;

			d_table.resize(n * r * c);
		}

		void insert(std::vector<ValueType> values) {
			DeviceVector<ValueType> d_values(values);

			insertKernel<<<BLOCKS, THREADS>>>(d_table.data(), d_table.size(), d_values.data(), d_values.size(), r);
			cudaDeviceSynchronize();
		}

		void peel() {
		}
	};

}

#endif
