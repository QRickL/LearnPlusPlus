#include "check.hpp"

void __lpp_check__(bool condition, const std::string& error_message)
{
    if (!condition) [[unlikely]] throw std::runtime_error(error_message);
}

void LPP::enforce_condition(bool condition, const std::string& error_message)
{
    if (!condition) [[unlikely]] throw std::runtime_error(error_message);
}