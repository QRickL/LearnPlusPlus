#ifndef LPP_VECTOR_OVERLOADS_HPP
#define LPP_VECTOR_OVERLOADS_HPP

#include <vector>

// Vector addition
std::vector<float> operator+(const std::vector<float>& v1, const std::vector<float>& v2);
std::vector<float> operator-(const std::vector<float>& v1, const std::vector<float>& v2);
std::vector<float> operator/(const std::vector<float>& v1, const std::vector<float>& v2);    // Division is defined by entry

// Vector dot product
float operator*(const std::vector<float>&v1, const std::vector<float>& v2);

// For adding biases
std::vector<float> operator+(const std::vector<float>& v1, float c);
std::vector<float> operator+(float c, const std::vector<float>& v2);

// Vector scalar multiplication
std::vector<float> operator*(const std::vector<float>& v1, float c);
std::vector<float> operator*(float c, const std::vector<float>& v2);

// Assignment operators
std::vector<float>& operator+=(std::vector<float>& v1, const std::vector<float>& v2);
std::vector<float>& operator+=(std::vector<float>& v1, const float c);
std::vector<float>& operator-=(std::vector<float>& v1, const std::vector<float>& v2);
std::vector<float>& operator*=(std::vector<float>& v1, const float c);
std::vector<float>& operator/=(std::vector<float>& v1, const std::vector<float>& v2);    // Division is defined by entry

namespace LPP {

// Fill operators
void set_all(std::vector<float>& v1, float s);

}

#endif