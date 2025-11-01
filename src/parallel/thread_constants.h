#include <thread>

namespace LPP {

const unsigned int MAX_THREADS = std::thread::hardware_concurrency();
const unsigned int USE_THREADS = std::max(1u, std::min(4u, MAX_THREADS));
const size_t PARALLEL_THRESHOLD = 1000000;

} // namespace LPP

// threshold = 100
// parallel         10166ms
// no parallel      6557ms

// threshold = 1000
// parallel         54956ms
// no parallel  