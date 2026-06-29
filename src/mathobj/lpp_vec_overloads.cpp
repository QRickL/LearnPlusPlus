#include "lpp_vec_overloads.h"
#include "check/check.h"
//#include "parallel/vector_parallel.h"

template <typename number>
LPP::Vect<number> operator+(const LPP::Vect<number>& v1, const LPP::Vect<number>& v2)  {
    LPP::Vect<number> result = v1;
    result += v2;
    return result;
}

template <typename number>
LPP::Vect<number> operator-(const LPP::Vect<number>& v1, const LPP::Vect<number>& v2) {
    LPP::Vect<number> result = v1;
    result -= v2;
    return result;
}

template <typename number>
number operator*(const LPP::Vect<number>& v1, const LPP::Vect<number>& v2) {
    _lpp_check_(v1.size() == v2.size(), "Dotting vectors of sizes " + std::to_string(v1.size()) + " and "  + std::to_string(v2.size()));

    number result = 0.0;
    for (size_t i = 0; i < v1.size(); i++) {
        result += v1[i]*v2[i];
    }
    return result;
}

template <typename number>
LPP::Vect<number> operator+(const LPP::Vect<number>& v1, number c) {
   LPP::Vect<number> result = v1;
   result += c;
   return result;
}

template <typename number>
LPP::Vect<number> operator+(number c, const LPP::Vect<number>& v2) {
    return v2 + c;
}

template <typename number>
LPP::Vect<number> operator*(const LPP::Vect<number>& v1, number c) {
    LPP::Vect<number> result = v1;
    result *= c;
    return result;
}

template <typename number>
LPP::Vect<number> operator*(number c, const LPP::Vect<number>& v2) {
    return v2 * c;
}

template <typename number>
LPP::Vect<number> operator/(const LPP::Vect<number>& v1, const LPP::Vect<number>& v2) {
    LPP::Vect<number> result = v1;
    result /= v2;
    return result;
}

/*
TODO: vectorization is currently disabled. I think there's a way to optimize it more...
TODO: check if this is hardware efficient
*/

template <typename number>
LPP::Vect<number>& operator+=(LPP::Vect<number>& v1, const LPP::Vect<number>& v2)
{
    _lpp_check_(v1.size() == v2.size(), "Adding vectors of sizes " + std::to_string(v1.size()) + " and "  + std::to_string(v2.size()));
    for (size_t i = 0; i < v1.size(); i++) v1[i] += v2[i];
    return v1;

    // if (v1.size() < LPP::CONSTANTS::VECTOR_PARALLEL_THRESHOLD) {
    //     for (size_t i = 0; i < v1.size(); i++) v1[i] += v2[i];
    // } else {
    //     LPP::parallel_comp(v1, v2, &LPP::add_helper);
    // }
    // return v1;
}

template <typename number>
LPP::Vect<number>& operator+=(LPP::Vect<number>& v1, number c)
{
    for (size_t i = 0; i < v1.size(); i++) v1[i] += c;
    return v1;

    // if (v1.size() < LPP::CONSTANTS::VECTOR_PARALLEL_THRESHOLD) {
    //     for (int i = 0; i < v1.size(); i++) v1[i] += c;
    // } else {
    //     LPP::parallel_comp(v1, c, &LPP::add_helper_scalar);
    // }
    // return v1;
}

template <typename number>
LPP::Vect<number>& operator-=(LPP::Vect<number>& v1, const LPP::Vect<number>& v2)
{
    _lpp_check_(v1.size() == v2.size(), "Subtracting vectors of sizes " + std::to_string(v1.size()) + " and "  + std::to_string(v2.size()));
    for (size_t i = 0; i < v1.size(); i++) v1[i] -= v2[i];
    return v1;

    // if (v1.size() < LPP::CONSTANTS::VECTOR_PARALLEL_THRESHOLD) {
    //     for (size_t i = 0; i < v1.size(); i++) v1[i] -= v2[i];
    // } else {
    //     LPP::parallel_comp(v1, v2, &LPP::sub_helper);
    // }
    // return v1;
}

template <typename number>
LPP::Vect<number>& operator*=(LPP::Vect<number>& v1, number c)
{
    for (size_t i = 0; i < v1.size(); i++) v1[i] *= c;
    return v1;

    // if (v1.size() < LPP::CONSTANTS::VECTOR_PARALLEL_THRESHOLD) {
    //     for (number& d : v1) d *= c;
    // } else {
    //     LPP::parallel_comp(v1, c, &LPP::mult_helper_scalar);
    // }
    // return v1;
}

template <typename number>
LPP::Vect<number>& operator/=(LPP::Vect<number>& v1, const LPP::Vect<number>& v2)
{
    _lpp_check_(v1.size() == v2.size(), "Division (by-entry) of vectors of sizes " + std::to_string(v1.size()) + " and "  + std::to_string(v2.size()));
    for (size_t i = 0; i < v1.size(); i++) v1[i] /= v2[i];
    return v1;

    // if (v1.size() < LPP::CONSTANTS::VECTOR_PARALLEL_THRESHOLD) {
    //     for (size_t i = 0; i < v1.size(); i++) v1[i] /= v2[i];
    // } else {
    //     LPP::parallel_comp(v1, v2, &LPP::div_helper);
    // }
    // return v1;
}