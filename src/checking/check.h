#ifndef LPP_CHECK_H
#define LPP_CHECK_H

#include <string>

// difference between the two is that we can turn off __lpp_check__ to save time during runtime if desired

void __lpp_check__(bool condition, const std::string& error_message);

namespace LPP {

void enforce_condition(bool condition, const std::string& error_message);

}

#endif
