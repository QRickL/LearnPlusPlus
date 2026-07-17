#!/bin/bash

CXXFLAGS="-std=c++20 -O3 -march=native -ffast-math"
OPENCV_CFLAGS="$(pkg-config --cflags opencv5)"
OPENCV_LIBS="-L$(brew --prefix opencv)/lib \
-lopencv_core \
-lopencv_imgproc \
-lopencv_imgcodecs"

# Library sources
g++ $CXXFLAGS -c ../../src/checking/check.cpp
g++ $CXXFLAGS -c ../../src/checking/timer.cpp -c
g++ $CXXFLAGS -c ../../src/functions/activations.cpp
g++ $CXXFLAGS -c ../../src/functions/losses.cpp
g++ $CXXFLAGS -c ../../src/functions/random_gen.cpp
g++ $CXXFLAGS -c ../../src/layer.cpp
g++ $CXXFLAGS -c ../../src/mathobj/matrix.cpp
g++ $CXXFLAGS -c ../../src/mathobj/vector_overloads.cpp
g++ $CXXFLAGS -c ../../src/network.cpp
g++ $CXXFLAGS -c ../../src/options/training_options.cpp
g++ $CXXFLAGS -c ../../src/options/classify.cpp
g++ $CXXFLAGS -c ../../src/regular/regularizers.cpp

# Main program
g++ $CXXFLAGS $OPENCV_CFLAGS -c brain.cpp

# Link everything
g++ $CXXFLAGS *.o -o brain.out $OPENCV_LIBS

# Cleanup
rm *.o

./brain.out