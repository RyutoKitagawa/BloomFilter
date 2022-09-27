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

#include "device_vector.cuh"

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

	__host__ __device__ inline unsigned int hash(ValueType value) {
		return (value * 0xDEADBEEF) >> 18;
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

	// NOTE: Expected number of duplicates should be relatively small (less than 200,000)
	__device__ void removeDuplicates(ValueType* values, unsigned int valueSize) {
		const unsigned int hashSize = 4096;
		__shared__ ValueType hashtable[hashSize];

		// Sets empty value for hash table to be 0xFFFFFFFF
		for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < hashSize; i += blockDim.x * gridDim.x)
			hashtable[i] = 0xFFFFFFFF;

		for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < valueSize; i += blockDim.x * gridDim.x) {
			unsigned int index = hash(values[i]);
			while (hashtable[index] != 0xFFFFFFFF && hashtable[index] != values[i]) {
				ValueType old = atmoicCAS(&(hashtable[index]), values[i], 0xFFFFFFFF);
				if (old == values[i]) break;
				index = (index < hashSize - 1) ? index + 1 : 0;
			}
		}

		__syncthreads();

		for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < hashSize; i += blockDim.x * gridDim.x) {
			values[i] = hashtable[i]
		}
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
			// Copy values to device
			DeviceVector<ValueType> d_values(values);

			// Insert values
			insertKernel<<<BLOCKS, THREADS>>>(d_table.data(), d_table.size(), d_values.data(), d_values.size(), r);
			cudaDeviceSynchronize();
		}

		void peel() {
		}
	};

}

#endif
