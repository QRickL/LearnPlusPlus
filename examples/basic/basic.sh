#!/bin/bash

g++ -std=c++20 -O3 -march=native -ffast-math ../../src/checking/check.cpp -c
g++ -std=c++20 -O3 -march=native -ffast-math ../../src/functions/activations.cpp -c
g++ -std=c++20 -O3 -march=native -ffast-math ../../src/functions/losses.cpp -c
g++ -std=c++20 -O3 -march=native -ffast-math ../../src/functions/random_gen.cpp -c
g++ -std=c++20 -O3 -march=native -ffast-math ../../src/layer.cpp -c
g++ -std=c++20 -O3 -march=native -ffast-math ../../src/mathobj/matrix.cpp -c
g++ -std=c++20 -O3 -march=native -ffast-math ../../src/mathobj/vector_overloads.cpp -c
g++ -std=c++20 -O3 -march=native -ffast-math ../../src/network.cpp -c
g++ -std=c++20 -O3 -march=native -ffast-math ../../src/options/training_options.cpp -c
# g++ -std=c++20 ../src/parallel/matrix_parallel.cpp -c
# g++ -std=c++20 ../src/parallel/vector_parallel.cpp -c
g++ -std=c++20 -O3 -march=native -ffast-math ../../src/regular/regularizers.cpp -c

g++ -std=c++20 -O3 -march=native -ffast-math basic.cpp -c

g++ -std=c++20 -O3 -march=native -ffast-math *.o -o basic.out
rm *.o

./basic.out