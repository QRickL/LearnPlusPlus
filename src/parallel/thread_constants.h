#ifndef LPP_THREAD_CONSTANTS_H
#define LPP_THREAD_CONSTANTS_H

#include <thread>

namespace LPP {

const unsigned int MAX_THREADS = std::thread::hardware_concurrency();
const unsigned int USE_THREADS = std::min(6u, MAX_THREADS);

const size_t MATRIX_PARALLEL_THRESHOLD = 7776;
const size_t VECTOR_PARALLEL_THRESHOLD = 46656;

}

#endif