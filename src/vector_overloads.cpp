#include "vector_overloads.h"
//#include "parallel/vector_parallel.h"
#include "checking/check.h"

std::vector<float> operator+(const std::vector<float>& v1, const std::vector<float>& v2) 
{
    std::vector<float> result = v1;
    result += v2;
    return result;
}

std::vector<float> operator-(const std::vector<float>& v1, const std::vector<float>& v2)
{
    std::vector<float> result = v1;
    result -= v2;
    return result;
}

float operator*(const std::vector<float>& v1, const std::vector<float>& v2)
{
    __lpp_check__(v1.size() == v2.size(), "operator* - dotting vectors of sizes " + std::to_string(v1.size()) + " and "  + std::to_string(v2.size()));

    float result = 0.f;
    for (size_t i = 0; i < v1.size(); i++) result += v1[i]*v2[i];
    return result;
}

std::vector<float> operator+(const std::vector<float>& v1, const float c)
{
   std::vector<float> result = v1;
   result += c;
   return result;
}

std::vector<float> operator+(const float c, const std::vector<float>& v2) {
    return v2 + c;
}

std::vector<float> operator*(const std::vector<float>& v1, const float c)
{
    std::vector<float> result = v1;
    result *= c;
    return result;
}

std::vector<float> operator*(const float c, const std::vector<float>& v2) {
    return v2 * c;
}

std::vector<float> operator/(const std::vector<float>& v1, const std::vector<float>& v2)
{
    std::vector<float> result = v1;
    result /= v2;
    return result;
}

std::vector<float>& operator+=(std::vector<float>& v1, const std::vector<float>& v2)
{
    __lpp_check__(v1.size() == v2.size(), "operator+ - adding vectors of sizes " + std::to_string(v1.size()) + " and "  + std::to_string(v2.size()));

    for (size_t i = 0; i < v1.size(); i++) v1[i] += v2[i];
    return v1;

    // if (v1.size() < LPP::CONSTANTS::VECTOR_PARALLEL_THRESHOLD) {
    //     for (size_t i = 0; i < v1.size(); i++) v1[i] += v2[i];
    // } else {
    //     LPP::parallel_comp(v1, v2, &LPP::add_helper);
    // }
    // return v1;
}

std::vector<float>& operator+=(std::vector<float>& v1, const float c)
{
    for (int i = 0; i < v1.size(); i++) v1[i] += c;
    return v1;
    
    // if (v1.size() < LPP::CONSTANTS::VECTOR_PARALLEL_THRESHOLD) {
    //     for (int i = 0; i < v1.size(); i++) v1[i] += c;
    // } else {
    //     LPP::parallel_comp(v1, c, &LPP::add_helper_scalar);
    // }
    // return v1;
}

std::vector<float>& operator-=(std::vector<float>& v1, const std::vector<float>& v2)
{
    __lpp_check__(v1.size() == v2.size(), "operator- - subtracting vectors of sizes " + std::to_string(v1.size()) + " and "  + std::to_string(v2.size()));
    for (size_t i = 0; i < v1.size(); i++) v1[i] -= v2[i];
    return v1;

    // if (v1.size() < LPP::CONSTANTS::VECTOR_PARALLEL_THRESHOLD) {
    //     for (size_t i = 0; i < v1.size(); i++) v1[i] -= v2[i];
    // } else {
    //     LPP::parallel_comp(v1, v2, &LPP::sub_helper);
    // }
    // return v1;
}

std::vector<float>& operator*=(std::vector<float>& v1, const float c)
{
    for (size_t i = 0; i < v1.size(); i++) v1[i] *= c;
    return v1;

    // if (v1.size() < LPP::CONSTANTS::VECTOR_PARALLEL_THRESHOLD) {
    //     for (float& d : v1) d *= c;
    // } else {
    //     LPP::parallel_comp(v1, c, &LPP::mult_helper_scalar);
    // }
    // return v1;
}

std::vector<float>& operator/=(std::vector<float>& v1, const std::vector<float>& v2)
{
    __lpp_check__(v1.size() == v2.size(), "operator/ - dividing vectors of sizes " + std::to_string(v1.size()) + " and "  + std::to_string(v2.size()));

    for (size_t i = 0; i < v1.size(); i++) v1[i] /= v2[i];
    return v1;

    // // Division by zero is the user's fault
    // if (v1.size() < LPP::CONSTANTS::VECTOR_PARALLEL_THRESHOLD) {
    //     for (size_t i = 0; i < v1.size(); i++) v1[i] /= v2[i];
    // } else {
    //     LPP::parallel_comp(v1, v2, &LPP::div_helper);
    // }
    // return v1;
}