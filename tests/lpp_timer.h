#ifndef LPP_TIMER_H
#define LPP_TIMER_H

#include <chrono>
#include <iostream>

namespace LPP
{

inline auto the_start = std::chrono::high_resolution_clock::now();
inline auto the_end = std::chrono::high_resolution_clock::now();

inline void start_timer() {
    the_start = std::chrono::high_resolution_clock::now();
}

inline void end_timer() {
    the_end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(the_end - the_start);
    std::cout << "Operation took: " << duration.count() << "ms" << std::endl;
}

} // namespace LPP

#endif