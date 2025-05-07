NVCC := nvcc
CC := gcc

NVCC_FLAGS := -O3 -arch=sm_89
CFLAGS := -O3

BIN := bin/
SRC := src/
LIB := include/

SOURCES := $(wildcard $(SRC)*.cu $(SRC)*.c)
LIBRARIES := $(wildcard $(LIB)*.h)
TMP_OBJECTS = $(SOURCES:$(SRC)%.c=$(BIN)%.o)
OBJECTS := $(TMP_OBJECTS:$(SRC)%.cu=$(BIN)%.o)

TARGET := $(BIN)spmv_three_way

all: $(TARGET)

debug_make:
	@echo "SOURCES=$(SOURCES)"
	@echo "LIBRARIES=$(LIBRARIES)"
	@echo "OBJECTS=$(OBJECTS)"

$(TARGET): $(SRC)spmv_three_way.cu $(MMIO_OBJ)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@

.PHONE: all clean

clean:
	rm -f $(BIN)*
