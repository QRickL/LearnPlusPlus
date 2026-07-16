#include "timer.hpp"

LPP::Timer::Timer() : now_{std::chrono::steady_clock::now()} {}

void LPP::Timer::click()
{
    auto temp = std::chrono::steady_clock::now();
    time_ = temp - now_;
    now_ = temp;
}
float LPP::Timer::time()
{
    return time_.count();
}