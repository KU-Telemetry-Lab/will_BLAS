# Location of the CUDA Toolkit
CUDA_PATH ?= /usr/local/cuda

SRCDIR = ./src
BINDIR = ./bin
OBJDIR = ./obj
INCDIR = ./inc
TESTDIR = ./test

$(shell   mkdir -p $(BINDIR))
$(shell   mkdir -p $(OBJDIR))

HOST_COMPILER ?= g++
NVCC          := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

# Gencode arguments
#  37  is best fit for K-80 chips
#  BUT,  with CUDA 11.0
#   nvcc warning : The 'compute_35', 'compute_37', 'compute_50', 'sm_35', 'sm_37' and 'sm_50'
#   architectures are deprecated, and may be removed in a future release
#   (Use -Wno-deprecated-gpu-targets to suppress warning).

# GENCODE_FLAGS += -gencode arch=compute_37,code=sm_37
# GENCODE_FLAGS += -Wno-deprecated-gpu-targets

# for the NVIDIA A100 GPU, 80 is the best fit.
GENCODE_FLAGS += -gencode arch=compute_80,code=sm_86

ifndef GENCODE_FLAGS
$(error Please uncomment appropriate GENCODE_FLAGS assignment in Makefile)
endif


# NVCC_FLAGS   := -std=c++11 -m64 $(GENCODE_FLAGS) --debug --generate-line-info -O2 --use_fast_math --forward-unknown-to-host-compiler -mavx2 --default-stream per-thread
NVCC_FLAGS   := -std=c++11 -m64 --debug --generate-line-info -O2 --use_fast_math --forward-unknown-to-host-compiler -mavx2 --default-stream per-thread
NVCC_LIBRARIES :=

$(foreach lf,$(LDFLAGS),$(eval LDFLAGS4NVCC +=  -Xcompiler \"$(lf)\"))
NVCC_LDFLAGS := $(LDFLAGS4NVCC)

SOURCES := $(wildcard $(SRCDIR)/*.cu)
OBJECTS := $(SOURCES:$(SRCDIR)/%.cu=$(OBJDIR)/%.o)

INC_DIRS := $(INCDIR)
# Include files add together a prefix, gcc make sense that -I flag
INC_FLAGS := $(addprefix -I,$(INC_DIRS))

# Make Makefiles output Dependency files
# That -MMD and -MP flags together to generate Makefiles
# That generated Makefiles will take .o as .d to the output
# That "-MMD" and "-MP" To generate the dependency files, all you have to do is
# add some flags to the compile command (supported by both Clang and GCC):
CPP_FLAGS ?= $(INC_FLAGS) -MMD -MP

# Dependency files
# To use the .d files, just need to find them all:
#
DEPS := $(OBJECTS:.o=.d)

################################################################################

ALL_EXES := matmult_0 matmult_1

MAIN_OBJECTS := $(foreach exe, $(ALL_EXES), $(OBJDIR)/$(exe).o)
NOT_MAIN_OBJECTS := $(filter-out $(MAIN_OBJECTS), $(OBJECTS))

# Target rules

.PHONY: all clean
all: $(ALL_EXES)

%.ptx: %.cu
	$(NVCC)  $(CPP_FLAGS) $(NVCC_FLAGS) -o $@ -ptx -src-in-ptx -c $<

$(OBJDIR)/%.o: $(SRCDIR)/%.cu
	$(NVCC)  $(CPP_FLAGS) $(NVCC_FLAGS) -o $@ -c $<

matmult_0: $(OBJDIR)/matmult_0.o $(NOT_MAIN_OBJECTS)
	$(NVCC) $(NVCC_FLAGS)  $(NVCC_LDFLAGS) -o $(BINDIR)/$@ $+ $(LIBRARIES) $(NVCC_LIBRARIES)

matmult_1: $(OBJDIR)/matmult_1.o $(NOT_MAIN_OBJECTS)
	$(NVCC) $(NVCC_FLAGS)  $(NVCC_LDFLAGS) -o $(BINDIR)/$@ $+ $(LIBRARIES) $(NVCC_LIBRARIES)

clean:
	rm -f $(OBJECTS)
	rm -f $(TESTDIR)/results/*.txt
	rm -f $(addprefix $(BINDIR)/, $(ALL_EXES)).
	rm -f $(OBJDIR)/*.d
