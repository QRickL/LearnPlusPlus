#pragma once
#include <string>

void __lpp_check__(bool condition, const std::string& error_message)
{
    if (!condition) throw std::runtime_error(error_message);
}
