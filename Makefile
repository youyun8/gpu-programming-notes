# Compile-check every solution. Does not require a GPU, only nvcc.
NVCC      ?= nvcc
ARCH      ?= sm_80
NVCCFLAGS ?= -O3 -std=c++17 -arch=$(ARCH) -Xcompiler -Wall

SOLUTIONS := $(shell find leetgpu tensara -name 'solution.cu' | sort)
OBJECTS   := $(patsubst %.cu,build/%.o,$(SOLUTIONS))

.PHONY: check test index site serve amd-isa clean

check: $(OBJECTS)
	@echo "compiled $(words $(OBJECTS)) solution(s)"

build/%.o: %.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

index:
	python3 scripts/build_index.py

# CPU emulator tests (needs clang++, torch; run scripts/fetch_upstream.sh first)
test:
	python3 tools/cuemu/run_tests.py --all

# Static site: build/site/ (needs requirements-docs.txt)
site:
	python3 scripts/build_site.py --strict --bundle
	mkdocs build --strict -f build/mkdocs.yml

serve:
	python3 scripts/build_site.py
	mkdocs serve -f build/mkdocs.yml

# ISA of the AMD tutorial kernel, using stock clang (no ROCm needed)
amd-isa:
	@mkdir -p build
	clang++ -x hip -nogpuinc -nogpulib --cuda-device-only --offload-arch=gfx942 -O3 -S \
	        -o build/mfma_gemm.s tutorials/amd/mfma_gemm.hip
	@echo "wrote build/mfma_gemm.s"

clean:
	rm -rf build
