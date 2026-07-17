#ifndef LPP_CHECK_HPP
#define LPP_CHECK_HPP

#include <string>

// Difference between the two is that we can turn off __lpp_check__ to save time during runtime if desired

#define LPP_ENABLE_CHECKS
#ifdef LPP_ENABLE_CHECKS

#define __lpp_check__(cond, msg)                     \
    do {                                            \
        if (!(cond))                                \
            throw std::runtime_error(msg);          \
    } while (0)

#else

#define __lpp_check__(cond, msg) ((void)0)

#endif

namespace LPP {

void enforce_condition(bool condition, const std::string& error_message);

}

#endif
