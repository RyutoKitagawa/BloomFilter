#ifndef DEVICE_VECTOR_CUH
#define DEVICE_VECTOR_CUH

#include <vector>

template<typename T>
class DeviceVector {
private:
	T *table;
	size_t n;

public:
	DeviceVector(size_t n = 0) {
		this->n = n;
		if (n > 0) cudaMalloc(&table, n * sizeof(T));
	}

	DeviceVector(std::vector<T> values) {
		this->n = values.size();
		if (values.size() > 0) cudaMalloc(&table, values.size() * sizeof(T));
		cudaMemcpy(table, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice);
	}

	~DeviceVector() {
		cudaFree(table);
	}

	void resize(size_t n) {
		if (n > 0) {
			cudaFree(table);
			cudaMalloc(&table, n * sizeof(T));
		}

		this->n = n;
	}

	T *data() const {
		return table;
	}

	size_t size() const {
		return n;
	}
};

#endif
