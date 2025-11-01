#ifndef LPP_VECTOR_OVERLOADS_S
#define LPP_VECTOR_OVERLOADS_S

#include <vector>

// Vector addition
std::vector<double> operator+(const std::vector<double>& v1, const std::vector<double>& v2);
std::vector<double> operator-(const std::vector<double>& v1, const std::vector<double>& v2);
std::vector<double> operator/(const std::vector<double>& v1, const std::vector<double>& v2);    // Division is defined by entry

// Vector dot product
double operator*(const std::vector<double>&v1, const std::vector<double>& v2);

// For adding biases
std::vector<double> operator+(const std::vector<double>& v1, double c);
std::vector<double> operator+(double c, const std::vector<double>& v2);

// Vector scalar multiplication
std::vector<double> operator*(const std::vector<double>& v1, double c);
std::vector<double> operator*(double c, const std::vector<double>& v2);

// Assignment operators
std::vector<double>& operator+=(std::vector<double>& v1, const std::vector<double>& v2);
std::vector<double>& operator+=(std::vector<double>& v1, const double c);
std::vector<double>& operator-=(std::vector<double>& v1, const std::vector<double>& v2);
std::vector<double>& operator*=(std::vector<double>& v1, const double c);
std::vector<double>& operator/=(std::vector<double>& v1, const std::vector<double>& v2);    // Division is defined by entry

#endif