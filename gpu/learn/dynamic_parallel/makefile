# compile each cpp file into separate executable

SHELL = /bin/sh

# CUDA directory:
CUDA_ROOT_DIR=/usr/local/cuda

##########################################################
## CXX COMPILER OPTIONS ##
CXX = g++
CXX_FLAGS = -std=c++11 -Wall -O3
CXX_LIBS =
CXX_INC =

##########################################################
## NVCC COMPILER OPTIONS ##
NVCC = nvcc
NVCC_FLAGS = -std=c++11 -O3 -arch=sm_61 -rdc=true
NVCC_LINK_FLAGS = -std=c++11 -O3 -arch=sm_61 -rdc=true -dlink
NVCC_LIBS =

# CUDA library directory:
CUDA_LIB_DIR= -L$(CUDA_ROOT_DIR)/lib64
# CUDA include directory:
CUDA_INC_DIR= -I$(CUDA_ROOT_DIR)/include
# CUDA linking libraries:
CUDA_LINK_LIBS= -lcudadevrt -lcudart


all: $(patsubst %.cu, %, $(wildcard *.cu))
%: %.o %_link.o
	${CXX} ${CXX_INC} ${CXX_FLAGS} $^ ${CUDA_INC_DIR} ${CUDA_LIB_DIR} ${CUDA_LINK_LIBS} -o $@.test

%.o: %.cu
	${NVCC} -c $(NVCC_FLAGS) $< -o $@ $(NVCC_LIBS)

%_link.o: %.o
	${NVCC}  $(NVCC_LINK_FLAGS) $< -o $@ $(CUDA_LINK_LIBS)

clean:
	-rm -f *.test *.o
