#include "vector_overloads.h"

std::vector<double> operator+(const std::vector<double>& v1, const std::vector<double>& v2) 
{
    // Dimension checking done by +=

    std::vector<double> result = v1;
    result += v2;
    return result;
}

std::vector<double> operator-(const std::vector<double>& v1, const std::vector<double>& v2)
{
    // Dimension checking done by -=

    std::vector<double> result = v1;
    result -= v2;
    return result;
}

double operator*(const std::vector<double>& v1, const std::vector<double>& v2)
{
    if (v1.size() != v2.size()) {
        const auto msg = "Dotting vectors of sizes " + std::to_string(v1.size()) + " and "  + std::to_string(v2.size());
        throw std::invalid_argument(msg);
    }

    double result = 0.0;
    for (size_t i = 0; i < v1.size(); i++) {
        result += v1[i]*v2[i];
    }
    return result;
}

// TODO: parallelize
std::vector<double> operator+(const std::vector<double>& v1, const double c)
{
    std::vector<double> result(v1.size());
    for (int i = 0; i < v1.size(); i++) {
        result[i] = c + v1[i];
    }
    return result;
}

std::vector<double> operator+(const double c, const std::vector<double>& v2) {
    return v2 + c;
}

std::vector<double> operator*(const std::vector<double>& v1, const double c)
{
    std::vector<double> result = v1;
    result *= c;
    return result;
}

std::vector<double> operator*(const double c, const std::vector<double>& v2) {
    return v2 * c;
}

// TODO: parallelize
std::vector<double> operator/(const std::vector<double>& v1, const std::vector<double>& v2)
{
    if (v1.size() != v2.size()) {
        const auto msg = "Entry-wise division of vectors of sizes " + std::to_string(v1.size()) + " and "  + std::to_string(v2.size());
        throw std::invalid_argument(msg);
    }

    std::vector<double> result(v1.size());
    for (size_t i = 0; i < v1.size(); i++) {
        result[i] = v1[i] / v2[i];
    }
    return result;
}

// TODO: parallelize
std::vector<double>& operator+=(std::vector<double>& v1, const std::vector<double>& v2)
{
    if (v1.size() != v2.size()) {
        const auto msg = "Adding vectors of sizes " + std::to_string(v1.size()) + " and "  + std::to_string(v2.size());
        throw std::invalid_argument(msg);
    }

    for (size_t i = 0; i < v1.size(); i++) {
        v1[i] += v2[i];
    }
    return v1;
}

// TODO: parallelize
std::vector<double>& operator-=(std::vector<double>& v1, const std::vector<double>& v2)
{
    if (v1.size() != v2.size()) {
        const auto msg = "Subtracting vectors of sizes " + std::to_string(v1.size()) + " and "  + std::to_string(v2.size());
        throw std::invalid_argument(msg);
    }

    for (size_t i = 0; i < v1.size(); i++) {
        v1[i] -= v2[i];
    }
    return v1;
}

// TODO: parallelize
std::vector<double>& operator*=(std::vector<double>& v1, const double c)
{
    for (double& d : v1) d *= c;
    return v1;
}