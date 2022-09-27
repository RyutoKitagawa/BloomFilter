#include "parallel_invertable_bloom_filter.cuh"

int main() {
	pibf::ParallelInvertableBloomFilter filter(5);

	std::vector<unsigned int> testArray = {12, 1, 3, 3, 1, 7, 7, 1, 2};
	filter.removeDuplicates(testArray);
}
