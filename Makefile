CARGO := cargo
CXX := g++

LIBTORCH ?= $(LIBTORCH)
CODON_PLL_A := ./lib/libcodon_pll.a
CODON_PLL_OBJ := ./lib/codon_pll.o

# Auto-detect whether the installed libtorch includes CUDA libraries.
# If libtorch_cuda.so exists, compile C++ code with -DUSE_CUDA.
HAS_CUDA_LIB := $(shell test -f $(LIBTORCH)/lib/libtorch_cuda.so && echo yes || echo no)
ifeq ($(HAS_CUDA_LIB),yes)
    CODON_PLL_FLAGS := -DUSE_CUDA
else
    CODON_PLL_FLAGS :=
endif

$(CODON_PLL_OBJ): ./codon_eval/codon_pll.cpp
	@mkdir -p $(dir $@)
	$(CXX) -c -O3 $< -o $@ -Iinclude -I$(LIBTORCH)/include -I$(LIBTORCH)/include/torch/csrc/api/include -std=c++17 $(CODON_PLL_FLAGS)

$(CODON_PLL_A): $(CODON_PLL_OBJ)
	ar rcs $@ $^

build: $(CODON_PLL_A)
	$(CARGO) build

release: $(CODON_PLL_A)
	$(CARGO) build --release

# CPU-only builds: compile codon_pll without CUDA support even if libtorch_cuda.so exists
$(CODON_PLL_OBJ:.o=_cpu.o): ./codon_eval/codon_pll.cpp
	@mkdir -p $(dir $@)
	$(CXX) -c -O3 $< -o $@ -Iinclude -I$(LIBTORCH)/include -I$(LIBTORCH)/include/torch/csrc/api/include -std=c++17

$(CODON_PLL_A:.a=_cpu.a): $(CODON_PLL_OBJ:.o=_cpu.o)
	ar rcs $@ $(CODON_PLL_OBJ:.o=_cpu.o)

build_cpu: $(CODON_PLL_A:.a=_cpu.a)
	CODON_CPU_ONLY=1 $(CARGO) build

release_cpu: $(CODON_PLL_A:.a=_cpu.a)
	CODON_CPU_ONLY=1 $(CARGO) build --release

# PLL-free builds: do not compile the C++ libtorch wrapper at all.
# The resulting binary is copied with a _no_pll suffix for easy identification.
build_no_pll:
	$(CARGO) build --no-default-features
	cp target/debug/cosyn target/debug/cosyn_no_pll

release_no_pll:
	$(CARGO) build --release --no-default-features
	cp target/release/cosyn target/release/cosyn_no_pll

clean:
	rm -f $(CODON_PLL_OBJ) $(CODON_PLL_A) $(CODON_PLL_OBJ:.o=_cpu.o) $(CODON_PLL_A:.a=_cpu.a)

.PHONY: clean build release build_cpu release_cpu build_no_pll release_no_pll
