#ifndef LPP_TIMER_H
#define LPP_TIMER_H

#include <chrono>
#include <iostream>

namespace LPP
{

auto timer_start = std::chrono::high_resolution_clock::now();
auto timer_end = std::chrono::high_resolution_clock::now();

void start_timer() {
    timer_start = std::chrono::high_resolution_clock::now();
}

void end_timer() {
    timer_end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(timer_end - timer_start);
    std::cout << duration.count() << "ms" << std::endl;
}

} // namespace LPP

#endif