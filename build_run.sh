rm malloc
nvcc malloc.cu -o malloc -L/usr/local/cuda/lib64/ -lcudart_static -ldl -lrt -rdc=true
./malloc -g 1 -b 1024 -n 8 > output.log