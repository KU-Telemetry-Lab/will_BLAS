# Location of the CUDA Toolkit
CUDA_PATH ?= /usr/local/cuda

SRCDIR = ./src
BINDIR = ./bin
OBJDIR = ./obj
INCDIR = ./inc
RESULTS = ./results

NVCC_FLAGS   := -std=c++11 -m64 --debug --generate-line-info -O2 --use_fast_math --forward-unknown-to-host-compiler -mavx2 --default-stream per-thread
NVCC_LIBRARIES := -L$(CUDA_PATH)/lib64

# Target rules
.PHONY: all clean

all: conv_0 conv_1 conv_2

conv_0:
	nvcc $(NVCC_FLAGS) -o $(BINDIR)/conv_0 $(SRCDIR)/naive/conv_0.cu $(SRCDIR)/naive/kernel_0.cu $(NVCC_LIBRARIES)

conv_1:
	nvcc $(NVCC_FLAGS) -o $(BINDIR)/conv_1 $(SRCDIR)/constant_memory/conv_1.cu $(SRCDIR)/constant_memory/kernel_1.cu $(NVCC_LIBRARIES)

conv_2:
	nvcc $(NVCC_FLAGS) -o $(BINDIR)/conv_2 $(SRCDIR)/shared_memory/conv_2.cu $(SRCDIR)/shared_memory/kernel_2.cu $(NVCC_LIBRARIES)

clean:
	rm -rf $(OBJDIR)/*.o $(RESULTS)/*.txt

