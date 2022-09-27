all: run_bf

d: bin/paribfd
	./bin/paribfd

run_bf: bin/paribf
	./bin/paribf

run_simulation:
	python src/simulation.py

bin/paribf: src/parallel_invertable_bloom_filter.cuh src/main.cu
	@mkdir -p bin
	nvcc -gencode arch=compute_86,code=sm_86 src/main.cu -o bin/paribf

bin/paribfd: src/parallel_invertable_bloom_filter.cuh src/main.cu
	@mkdir -p bin
	nvcc -DDEBUG -gencode arch=compute_86,code=sm_86 src/main.cu -o bin/paribfd
