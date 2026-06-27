#ifndef LPP_Vec_OVERLOADS_S
#define LPP_Vec_OVERLOADS_S

#include "data_types.h"
#include <vector>

// Returns a new vector

LPP_Vec operator+(const LPP_Vec& v1, const LPP_Vec& v2);    // Vector addition
LPP_Vec operator+(const LPP_Vec& v1, LPP_Num c);            // Vector scalar addition
LPP_Vec operator+(LPP_Num c, const LPP_Vec& v2);            // Vector scalar addition
LPP_Vec operator-(const LPP_Vec& v1, const LPP_Vec& v2);    // Vector subtraction
LPP_Num operator*(const LPP_Vec&v1, const LPP_Vec& v2);     // Vector dot product
LPP_Vec operator*(const LPP_Vec& v1, LPP_Num c);            // Vector scalar multiplication
LPP_Vec operator*(LPP_Num c, const LPP_Vec& v2);            // Vector scalar multiplication
LPP_Vec operator/(const LPP_Vec& v1, const LPP_Vec& v2);    // Vector division element-wise; division by 0 is user's fault


// Modifies existing LHS vector

LPP_Vec& operator+=(LPP_Vec& v1, const LPP_Vec& v2);
LPP_Vec& operator+=(LPP_Vec& v1, LPP_Num c);
LPP_Vec& operator-=(LPP_Vec& v1, const LPP_Vec& v2);
LPP_Vec& operator*=(LPP_Vec& v1, LPP_Num c);
LPP_Vec& operator/=(LPP_Vec& v1, const LPP_Vec& v2);

#endif