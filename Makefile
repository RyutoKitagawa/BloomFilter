all: toy

d: bin/paribfd
	./bin/paribfd

run_bf: bin/paribf
	./bin/paribf

run_simulation:
	python src/simulation.py

toy: bin/toy
	./bin/toy

bin/paribf: src/parallel_invertable_bloom_filter.cuh src/main.cu
	@mkdir -p bin
	nvcc -gencode arch=compute_86,code=sm_86 src/main.cu -o bin/paribf

bin/paribfd: src/parallel_invertable_bloom_filter.cuh src/main.cu
	@mkdir -p bin
	nvcc -DDEBUG -gencode arch=compute_86,code=sm_86 src/main.cu -o bin/paribfd

bin/toy: src/toy.cu
	@mkdir -p bin
	nvcc -gencode arch=compute_86,code=sm_86 src/toy.cu -o bin/toy
