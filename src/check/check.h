#ifndef LPP_CHECK_H
#define LPP_CHECK_H

#include "string"

void _lpp_check_(bool condition, const std::string& msg)
{
    if (!condition) [[unlikely]] {throw std::invalid_argument(msg);}
}

#endif