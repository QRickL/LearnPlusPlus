#pragma once

#include "lpp_vec.h"

namespace LPP {

// Following will construct a new vector:

template <typename number>
Vect<number> operator+(const Vect<number>& v1, const Vect<number>& v2);

template <typename number>
Vect<number> operator+(const Vect<number>& v1, number c);

template <typename number>
Vect<number> operator+(number c, const Vect<number>& v2);

template <typename number>
Vect<number> operator-(const Vect<number>& v1, const Vect<number>& v2);

template <typename number>
number operator*(const Vect<number>&v1, const Vect<number>& v2);

template <typename number>
Vect<number> operator*(const Vect<number>& v1, number c);

template <typename number>
Vect<number> operator*(number c, const Vect<number>& v2);

template <typename number>
Vect<number> operator/(const Vect<number>& v1, const Vect<number>& v2);


// Following will edit existing vector:

template <typename number>
Vect<number>& operator+=(Vect<number>& v1, const Vect<number>& v2);

template <typename number>
Vect<number>& operator+=(Vect<number>& v1, number c);

template <typename number>
Vect<number>& operator-=(Vect<number>& v1, const Vect<number>& v2);

template <typename number>
Vect<number>& operator*=(Vect<number>& v1, number c);

template <typename number>
Vect<number>& operator/=(Vect<number>& v1, const Vect<number>& v2);

}