# Compile-check every solution and tutorial program. Does not require a GPU, only nvcc.
NVCC      ?= nvcc
ARCH      ?= sm_80
NVCCFLAGS ?= -O3 -std=c++17 -arch=$(ARCH) -Xcompiler -Wall

SOLUTIONS := $(shell find leetgpu tensara -name 'solution.cu' | sort)
OBJECTS   := $(patsubst %.cu,build/%.o,$(SOLUTIONS))
GEMM      := $(sort $(wildcard tutorials/gemm/*.cu))
GEMM_OBJS := $(patsubst %.cu,build/%.o,$(GEMM))
EXAMPLES  := $(sort $(wildcard tutorials/examples/*.cu))
EX_OBJS   := $(patsubst %.cu,build/%.o,$(EXAMPLES))

.PHONY: check test gemm-test examples-test triton-test figures index site serve amd-isa clean

check: $(OBJECTS) $(GEMM_OBJS) $(EX_OBJS)
	@echo "compiled $(words $(OBJECTS)) solution(s), $(words $(GEMM_OBJS)) GEMM and $(words $(EX_OBJS)) example program(s)"

build/tutorials/gemm/%.o: tutorials/gemm/%.cu tutorials/gemm/harness.cuh
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

build/tutorials/examples/%.o: tutorials/examples/%.cu tutorials/examples/check.cuh
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

build/%.o: %.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Run every GEMM tutorial program's --test mode on the CPU emulator (needs clang++ only),
# plus the split-K and Stream-K variants.
CUEMU_RUN := python3 tools/cuemu/cuemu.py run
gemm-test:
	@set -e; for f in $(GEMM); do $(CUEMU_RUN) $$f -- --test; done
	@$(CUEMU_RUN) tutorials/gemm/06-split-k.cu -- --atomic --test
	@$(CUEMU_RUN) tutorials/gemm/06-split-k.cu -- --splits=5 --test
	@set -e; for g in 1 3 7 13; do $(CUEMU_RUN) tutorials/gemm/07-stream-k.cu -- --blocks=$$g --test; done
	@CUEMU_REVERSE=1 $(CUEMU_RUN) tutorials/gemm/09-mma-sync.cu -- --test

# Run the example programs of chapters 09-13 on the CPU emulator (their default mode checks results).
examples-test:
	@set -e; for f in $(EXAMPLES); do $(CUEMU_RUN) $$f; done

# The Triton kernels of chapter 14 (needs torch and triton; uses the interpreter without a GPU).
triton-test:
	cd tutorials/examples/14-triton && python3 test_kernels.py

# Regenerate the tutorial figures (tutorials/figures/*.svg) from scripts/figures/
figures:
	python3 scripts/build_figures.py
	python3 scripts/check_figures.py

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
