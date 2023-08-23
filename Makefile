TARGET = poisson_solver

# Directory paths
SRCDIR = src
INCDIR = include
BINDIR = bin
OBJDIR = obj

# List of source files and object files
SOURCES = $(SRCDIR)/main.cu $(SRCDIR)/print.c $(SRCDIR)/alloc3d.c $(SRCDIR)/initialize.c $(SRCDIR)/alloc3d_device.c $(SRCDIR)/alloc3d_cuda.cu $(SRCDIR)/jacobi_cuda.cu
OBJECTS = $(OBJDIR)/print.o $(OBJDIR)/alloc3d.o $(OBJDIR)/initialize.o $(OBJDIR)/alloc3d_device.o $(OBJDIR)/alloc3d_cuda.o 

MAIN = $(OBJDIR)/main.o
OBJS = $(MAIN) $(OBJDIR)/jacobi_ps.o $(OBJDIR)/jacobi_offload.o $(OBJDIR)/jacobi_cuda.o

# options and settings for the GCC compilers
CC = mpic++
CXX = nvc++

OPT = -g -fast -Msafeptr -Minfo -mp=gpu -gpu=pinned -gpu=cc70 -gpu=lineinfo -mp=noautopar -cuda
#PIC   = -fpic -shared
ISA = 
PARA = -fopenmp
CUDA_PATH ?= /appl/cuda/12.1.0
INC = -I$(CUDA_PATH)/include -I/appl/nvhpc/2023_231/Linux_x86_64/23.1/examples/OpenMP/SDK/include -I$(MODULE_MPI_INCLUDE_DIR) -I/appl/nccl/2.17.1-1-cuda-12.1/include -I$(INCDIR)
LIBS = -lcuda -lnccl
LDFLAGS = -lm -L/appl/nccl/2.17.1-1-cuda-12.1/lib -lnccl

CXXFLAGS = $(OPT) $(PIC) $(INC) $(ISA) $(PARA) $(XOPT)
CFLAGS = $(OPT) $(PIC) $(INC) $(ISA) $(PARA) $(XOPT)

all: $(BINDIR)/$(TARGET)

$(BINDIR)/$(TARGET): $(OBJECTS) $(OBJS)
	$(CC) -o $@ $(CFLAGS) $(OBJS) $(OBJECTS) $(LDFLAGS)

# Pattern rule for .c files
$(OBJDIR)/%.o: $(SRCDIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

# Pattern rule for .cu files
$(OBJDIR)/%.o: $(SRCDIR)/%.cu
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	@/bin/rm -f $(OBJDIR)/*.o *~

realclean: clean
	@/bin/rm -f $(BINDIR)/$(TARGET)

# DO NOT DELETE
main.o: $(SRCDIR)/main.cu $(INCDIR)/print.h $(INCDIR)/jacobi_offload.h $(INCDIR)/initialize
