CC = g++
OPENCL ?= 0
OPENCL_INCLUDE ?=
OPENCL_LIBS ?= -lOpenCL
COMPILER_FLAGS = -Wall -O3 -std=c++23 -pedantic -Werror=format-security -flto=auto -ftree-loop-vectorize -pthread $(OPENCL_INCLUDE)
LINKER_FLAGS = -I../src
LDLIBS =
ifeq ($(OPENCL),1)
COMPILER_FLAGS += -DZYX_USE_OPENCL
LDLIBS += $(OPENCL_LIBS)
endif

.PHONY: all clean

all: test_buffer test_tensor test_activations test_losses test_layers test_models

test_buffer: test_buffer.cpp
	$(CC) $(COMPILER_FLAGS) $(LINKER_FLAGS) -o $@ $^ $(LDLIBS)

test_tensor: test_tensor.cpp
	$(CC) $(COMPILER_FLAGS) $(LINKER_FLAGS) -o $@ $^ $(LDLIBS)

test_activations: test_activations.cpp
	$(CC) $(COMPILER_FLAGS) $(LINKER_FLAGS) -o $@ $^ $(LDLIBS)

test_losses: test_losses.cpp
	$(CC) $(COMPILER_FLAGS) $(LINKER_FLAGS) -o $@ $^ $(LDLIBS)

test_layers: test_layers.cpp
	$(CC) $(COMPILER_FLAGS) $(LINKER_FLAGS) -o $@ $^ $(LDLIBS)

test_models: test_models.cpp
	$(CC) $(COMPILER_FLAGS) $(LINKER_FLAGS) -o $@ $^ $(LDLIBS)

clean:
	rm -f test_buffer test_tensor test_activations test_losses test_layers test_models
