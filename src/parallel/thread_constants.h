#include <thread>

namespace LPP {

const unsigned int MAX_THREADS = std::thread::hardware_concurrency();
const unsigned int USE_THREADS = std::min(6u, MAX_THREADS);

//const size_t MATRIX_PARALLEL_THRESHOLD = 1000;
const size_t VECOTR_PARALLEL_THRESHOLD = 50000;

}