#include "vector_overloads.h"
#include "parallel/vector_parallel.h"

std::vector<double> operator+(const std::vector<double>& v1, const std::vector<double>& v2) 
{
    std::vector<double> result = v1;
    result += v2;
    return result;
}

std::vector<double> operator-(const std::vector<double>& v1, const std::vector<double>& v2)
{
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

std::vector<double> operator+(const std::vector<double>& v1, const double c)
{
   std::vector<double> result = v1;
   result += c;
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

std::vector<double> operator/(const std::vector<double>& v1, const std::vector<double>& v2)
{
    std::vector<double> result = v1;
    result /= v2;
    return result;
}

std::vector<double>& operator+=(std::vector<double>& v1, const std::vector<double>& v2)
{
    if (v1.size() != v2.size()) {
        const auto msg = "Adding vectors of sizes " + std::to_string(v1.size()) + " and "  + std::to_string(v2.size());
        throw std::invalid_argument(msg);
    }

    if (v1.size() < LPP::VECTOR_PARALLEL_THRESHOLD) {
        for (size_t i = 0; i < v1.size(); i++) v1[i] += v2[i];
    } else {
        LPP::parallel_comp(v1, v2, &LPP::add_helper);
    }
    return v1;
}

std::vector<double>& operator+=(std::vector<double>& v1, const double c)
{
    if (v1.size() < LPP::VECTOR_PARALLEL_THRESHOLD) {
        for (int i = 0; i < v1.size(); i++) v1[i] += c;
    } else {
        LPP::parallel_comp(v1, c, &LPP::add_helper_scalar);
    }
    return v1;
}

std::vector<double>& operator-=(std::vector<double>& v1, const std::vector<double>& v2)
{
    if (v1.size() != v2.size()) {
        const auto msg = "Subtracting vectors of sizes " + std::to_string(v1.size()) + " and "  + std::to_string(v2.size());
        throw std::invalid_argument(msg);
    }

    if (v1.size() < LPP::VECTOR_PARALLEL_THRESHOLD) {
        for (size_t i = 0; i < v1.size(); i++) v1[i] -= v2[i];
    } else {
        LPP::parallel_comp(v1, v2, &LPP::sub_helper);
    }
    return v1;
}

std::vector<double>& operator*=(std::vector<double>& v1, const double c)
{
    if (v1.size() < LPP::VECTOR_PARALLEL_THRESHOLD) {
        for (double& d : v1) d *= c;
    } else {
        LPP::parallel_comp(v1, c, &LPP::mult_helper_scalar);
    }
    return v1;
}

std::vector<double>& operator/=(std::vector<double>& v1, const std::vector<double>& v2)
{
    if (v1.size() != v2.size()) {
        const auto msg = "Division (by-entry) of vectors of sizes " + std::to_string(v1.size()) + " and "  + std::to_string(v2.size());
        throw std::invalid_argument(msg);
    }

    // Division by zero is the user's fault
    if (v1.size() < LPP::VECTOR_PARALLEL_THRESHOLD) {
        for (size_t i = 0; i < v1.size(); i++) v1[i] /= v2[i];
    } else {
        LPP::parallel_comp(v1, v2, &LPP::div_helper);
    }
    return v1;
}