CC = g++
OPENCL ?= 0
OPENCL_INCLUDE ?=
OPENCL_LIBS ?= -lOpenCL
COMPILER_FLAGS = -Wall -std=c++23 -pedantic -Werror=format-security -flto=auto -ftree-loop-vectorize -Wshadow-compatible-local -O3 $(OPENCL_INCLUDE)
LINKER_FLAGS = -I../src
LDLIBS =
ifeq ($(OPENCL),1)
COMPILER_FLAGS += -DZYX_USE_OPENCL
LDLIBS += $(OPENCL_LIBS)
endif

.PHONY: all clean

all: example1 example2 network_example1 network_example2

example1: example1.cpp
	$(CC) $(COMPILER_FLAGS) $(LINKER_FLAGS) -o $@ $^ $(LDLIBS)

example2: example2.cpp
	$(CC) $(COMPILER_FLAGS) $(LINKER_FLAGS) -o $@ $^ $(LDLIBS)

network_example1: network_example1.cpp
	$(CC) $(COMPILER_FLAGS) $(LINKER_FLAGS) -o $@ $^ $(LDLIBS)

network_example2: network_example2.cpp
	$(CC) $(COMPILER_FLAGS) $(LINKER_FLAGS) -o $@ $^ $(LDLIBS)

clean:
	rm -f example1 example2 network_example1 network_example2
	find . -name "*.o" -type f -delete;
