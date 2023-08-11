# Makefile
#
TARGET = poisson_solver

SOURCES	= main.cu print.c alloc3d.c initialize.c alloc3d_device.c alloc3d_cuda.cu jacobi_cuda.cu
OBJECTS	= print.o alloc3d.o initialize.o alloc3d_device.o alloc3d_cuda.o 

MAIN = main.o
OBJS = $(MAIN) jacobi_ps.o jacobi_offload.o jacobi_cuda.o

# options and settings for the GCC compilers
CC	= mpic++
CXX	= nvc++

OPT	= -g -fast -Msafeptr -Minfo -mp=gpu -gpu=pinned -gpu=cc70 -gpu=lineinfo -mp=noautopar -cuda
#PIC   = -fpic -shared
ISA	= 
PARA	= -fopenmp
CUDA_PATH ?= /appl/cuda/12.1.0
INC   = -I$(CUDA_PATH)/include -I/appl/nvhpc/2023_231/Linux_x86_64/23.1/examples/OpenMP/SDK/include -I$(MODULE_MPI_INCLUDE_DIR) -I/appl/nccl/2.17.1-1-cuda-12.1/include
LIBS	= -lcuda -lnccl
LDFLAGS = -lm -L/appl/nccl/2.17.1-1-cuda-12.1/lib -lnccl

CXXFLAGS= $(OPT) $(PIC) $(INC) $(ISA) $(PARA) $(XOPT)
CFLAGS= $(OPT) $(PIC) $(INC) $(ISA) $(PARA) $(XOPT)

all: $(TARGET)

$(TARGET): $(OBJECTS) $(OBJS)
	$(CC) -o $@ $(CFLAGS) $(OBJS) $(OBJECTS) $(LDFLAGS)

$(MAIN):
	$(CC) -o $@ $(CFLAGS) -c main.cu alloc3d_cuda.o jacobi_cuda.o

alloc3d_cuda.o: alloc3d_cuda.cu
	$(CC) -o $@ $(CFLAGS) -c $<

jacobi_cuda.o: jacobi_cuda.cu
	$(CC) -o $@ $(CFLAGS) -c $<

clean:
	@/bin/rm -f core *.o *~

realclean: clean
	@/bin/rm -f $(TARGET)

# DO NOT DELETE
main.o: main.cu print.h jacobi_offload.h initialize.h jacobi_cuda.hpp
print.o: print.h
