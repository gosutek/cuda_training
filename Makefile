NVCC = nvcc
CC = gcc

NVCC_FLAGS = -O3 -arch=sm_89
CFLAGS = -O3

TARGETS = vector_add iterate_over_2d iterate_over_3d spmv_three_way

SRCS_VECTOR_ADD = vector_add.cu
SRCS_ITERATE_2D = iterate_over_2d.cu
SRCS_ITERATE_3D = iterate_over_3d.cu
SRCS_SPMV_THREE_WAY = spmv_three_way.cu

OBJS_VECTOR_ADD = $(SRCS_VECTOR_ADD:.cu=.o)
OBJS_ITERATE_2D = $(SRCS_ITERATE_2D:.cu=.o)
OBJS_ITERATE_3D = $(SRCS_ITERATE_3D:.cu=.o)
OBJS_SPMV_THREE_WAY = $(SRCS_SPMV_THREE_WAY:.cu=.o) lib/mmio.o

all: $(TARGETS)

vector_add: $(OBJS_VECTOR_ADD)
	$(NVCC) $(NVCC_FLAGS) $< -o $@

iterate_over_2d: $(OBJS_ITERATE_2D)
	$(NVCC) $(NVCC_FLAGS) $< -o $@

iterate_over_3d: $(OBJS_ITERATE_3D)
	$(NVCC) $(NVCC_FLAGS) $< -o $@

spmv_three_way: $(OBJS_SPMV_THREE_WAY)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

lib/mmio.o: lib/mmio.c lib/mmio.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS_VECTOR_ADD) $(OBJS_ITERATE_2D) $(OBJS_ITERATE_3D) $(OBJS_SPMV_THREE_WAY) lib/mmio.o $(TARGETS)

.PHONY: all clean
