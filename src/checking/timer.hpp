#ifndef LPP_TIMER_HPP
#define LPP_TIMER_HPP

#include <chrono>

namespace LPP {

class Timer {
    std::__1::chrono::steady_clock::time_point now_;
    std::chrono::duration<float, std::milli> time_;

public:
    Timer();
    void click();
    float time();

};

}

#endif
