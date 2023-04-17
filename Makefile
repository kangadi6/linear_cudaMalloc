# Define the compiler and compiler flags
NVCC = nvcc
# NVCCFLAGS = -O2 -arch=sm_70

# Define the name of the executable
EXE = malloc

# Define the list of source files
SRC = malloc.cu

# Define the list of object files
OBJ = $(SRC:.cu=.o)

# Define the default target
all: $(EXE)

# Define the rule for linking the object files into the executable
$(EXE): $(SRC)
	$(NVCC) $(SRC) -o $@ -rdc=true

# Define the clean target
clean:
	rm -f $(OBJ) $(EXE)
