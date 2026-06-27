#include "vector_overloads.h"
#include "parallel/vector_parallel.h"
#include "check/check.h"

LPP_Vec operator+(const LPP_Vec& v1, const LPP_Vec& v2)  {
    LPP_Vec result = v1;
    result += v2;
    return result;
}

LPP_Vec operator-(const LPP_Vec& v1, const LPP_Vec& v2) {
    LPP_Vec result = v1;
    result -= v2;
    return result;
}

LPP_Num operator*(const LPP_Vec& v1, const LPP_Vec& v2) {
    _lpp_check_(v1.size() == v2.size(), "Dotting vectors of sizes " + std::to_string(v1.size()) + " and "  + std::to_string(v2.size()));

    LPP_Num result = 0.0;
    for (size_t i = 0; i < v1.size(); i++) {
        result += v1[i]*v2[i];
    }
    return result;
}

LPP_Vec operator+(const LPP_Vec& v1, LPP_Num c) {
   LPP_Vec result = v1;
   result += c;
   return result;
}

LPP_Vec operator+(LPP_Num c, const LPP_Vec& v2) {
    return v2 + c;
}

LPP_Vec operator*(const LPP_Vec& v1, LPP_Num c) {
    LPP_Vec result = v1;
    result *= c;
    return result;
}

LPP_Vec operator*(LPP_Num c, const LPP_Vec& v2) {
    return v2 * c;
}

LPP_Vec operator/(const LPP_Vec& v1, const LPP_Vec& v2) {
    LPP_Vec result = v1;
    result /= v2;
    return result;
}

/*
TODO: vectorization is currently disabled. I think there's a way to optimize it more...
TODO: check if this is hardware efficient
*/

LPP_Vec& operator+=(LPP_Vec& v1, const LPP_Vec& v2)
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

LPP_Vec& operator+=(LPP_Vec& v1, LPP_Num c)
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

LPP_Vec& operator-=(LPP_Vec& v1, const LPP_Vec& v2)
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

LPP_Vec& operator*=(LPP_Vec& v1, LPP_Num c)
{
    for (size_t i = 0; i < v1.size(); i++) v1[i] *= c;
    return v1;

    // if (v1.size() < LPP::CONSTANTS::VECTOR_PARALLEL_THRESHOLD) {
    //     for (LPP_Num& d : v1) d *= c;
    // } else {
    //     LPP::parallel_comp(v1, c, &LPP::mult_helper_scalar);
    // }
    // return v1;
}

LPP_Vec& operator/=(LPP_Vec& v1, const LPP_Vec& v2)
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