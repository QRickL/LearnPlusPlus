#!/bin/bash

g++ ../src/checking/check.cpp -c
g++ ../src/functions/losses.cpp -c
g++ ../src/functions/activations.cpp -c
g++ ../src/functions/random_gen.cpp -c

# g++ ../src/parallel/vector_parallel.cpp -c
# g++ ../src/parallel/matrix_parallel.cpp -c

g++ ../src/mathobj/matrix.cpp -c
g++ ../src/layer.cpp -c
g++ ../src/network.cpp -c
g++ ../src/mathobj/vector_overloads.cpp -c
g++ ../src/regular/regularizers.cpp -c
g++ ../src/options/training_options.cpp -c

g++ basic.cpp -c

g++ *.o -o basic.out
rm *.o

./basic.out