#pragma once

#include <iostream>
#include "../mathobj/lpp_vec.h"
#include "../matrix.h"
#include "../layer.h"

namespace LPP {

// Display matrix entries
template <typename number>
void printObject(const Matrix<number>& m, std::ostream& os = std::cout);

// Display vector entries
template <typename number>
void printObject(const Vect<number>& v, std::ostream& os = std::cout);

template <typename number>
void printObject(const Layer<number>& l, std::ostream& os = std::cout);

}