#include "../src/LPP.h"
#include "lpp_timer.h"

using namespace LPP;

int main() {

    start_timer();

    for (int i = 0; i <= 14; i++) {
        const int s = 1 << i;
        LPP::Matrix m1(std::vector<std::vector<double>>(s, std::vector<double>(s)));
        LPP::Matrix m2(std::vector<std::vector<double>>(s, std::vector<double>(s)));

        std::cout << s << ": addition" << std::endl;;
        LPP::start_timer();
        m1 -= m2;
        LPP::end_timer();

        std::cout << s << ": scalar mult" << std::endl;;
        LPP::start_timer();
        m1 *= 1;
        LPP::end_timer();
        std::cout << std::endl;
    }

    end_timer();

}

// Vectorization:
/*
1: addition
Operation took: 0ms
1: scalar mult
Operation took: 0ms

2: addition
Operation took: 0ms
2: scalar mult
Operation took: 0ms

4: addition
Operation took: 0ms
4: scalar mult
Operation took: 0ms

8: addition
Operation took: 0ms
8: scalar mult
Operation took: 0ms

16: addition
Operation took: 0ms
16: scalar mult
Operation took: 0ms

32: addition
Operation took: 0ms
32: scalar mult
Operation took: 0ms

64: addition
Operation took: 0ms
64: scalar mult
Operation took: 0ms

128: addition
Operation took: 0ms
128: scalar mult
Operation took: 0ms

256: addition
Operation took: 0ms
256: scalar mult
Operation took: 0ms

512: addition
Operation took: 0ms
512: scalar mult
Operation took: 0ms

1024: addition
Operation took: 0ms
1024: scalar mult
Operation took: 1ms

2048: addition
Operation took: 2ms
2048: scalar mult
Operation took: 3ms

4096: addition
Operation took: 9ms
4096: scalar mult
Operation took: 14ms

8192: addition
Operation took: 47ms
8192: scalar mult
Operation took: 55ms

16384: addition
Operation took: 541ms
16384: scalar mult
Operation took: 231ms

Operation took: 1991ms
*/

// No vectorization
/*
1: addition
Operation took: 0ms
1: scalar mult
Operation took: 0ms

2: addition
Operation took: 0ms
2: scalar mult
Operation took: 0ms

4: addition
Operation took: 0ms
4: scalar mult
Operation took: 0ms

8: addition
Operation took: 0ms
8: scalar mult
Operation took: 0ms

16: addition
Operation took: 0ms
16: scalar mult
Operation took: 0ms

32: addition
Operation took: 0ms
32: scalar mult
Operation took: 0ms

64: addition
Operation took: 0ms
64: scalar mult
Operation took: 0ms

128: addition
Operation took: 0ms
128: scalar mult
Operation took: 0ms

256: addition
Operation took: 0ms
256: scalar mult
Operation took: 0ms

512: addition
Operation took: 1ms
512: scalar mult
Operation took: 1ms

1024: addition
Operation took: 4ms
1024: scalar mult
Operation took: 6ms

2048: addition
Operation took: 14ms
2048: scalar mult
Operation took: 19ms

4096: addition
Operation took: 58ms
4096: scalar mult
Operation took: 76ms

8192: addition
Operation took: 216ms
8192: scalar mult
Operation took: 298ms

16384: addition
Operation took: 1677ms
16384: scalar mult
Operation took: 1250ms

Operation took: 2994ms
*/