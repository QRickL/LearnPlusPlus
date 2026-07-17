#include "check.hpp"

void LPP::enforce_condition(bool condition, const std::string& error_message)
{
    if (!condition) [[unlikely]] throw std::runtime_error(error_message);
}