# Compile-check every solution. Does not require a GPU, only nvcc.
NVCC      ?= nvcc
ARCH      ?= sm_80
NVCCFLAGS ?= -O3 -std=c++17 -arch=$(ARCH) -Xcompiler -Wall

SOLUTIONS := $(shell find leetgpu tensara -name 'solution.cu' | sort)
OBJECTS   := $(patsubst %.cu,build/%.o,$(SOLUTIONS))

.PHONY: check index clean

check: $(OBJECTS)
	@echo "compiled $(words $(OBJECTS)) solution(s)"

build/%.o: %.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

index:
	python3 scripts/build_index.py

clean:
	rm -rf build
