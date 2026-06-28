#pragma once

#include <iostream>
#include "../mathobj/lpp_vec.h"
#include "../matrix.h"

namespace LPP {

// Display matrix entries
template <typename number>
void print_object(const Matrix<number>& m, std::ostream& os = std::cout);

// Display vector entries
template <typename number>
void print_object(const Vect<number>& v, std::ostream& os = std::cout);

}