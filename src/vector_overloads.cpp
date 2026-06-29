#include "vector_overloads.h"
#include "parallel/vector_parallel.h"

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
    if (v1.size() != v2.size()) {
        const auto msg = "Dotting vectors of sizes " + std::to_string(v1.size()) + " and "  + std::to_string(v2.size());
        throw std::invalid_argument(msg);
    }

    float result = 0.0;
    for (size_t i = 0; i < v1.size(); i++) {
        result += v1[i]*v2[i];
    }
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
    if (v1.size() != v2.size()) {
        const auto msg = "Adding vectors of sizes " + std::to_string(v1.size()) + " and "  + std::to_string(v2.size());
        throw std::invalid_argument(msg);
    }

    if (v1.size() < LPP::CONSTANTS::VECTOR_PARALLEL_THRESHOLD) {
        for (size_t i = 0; i < v1.size(); i++) v1[i] += v2[i];
    } else {
        LPP::parallel_comp(v1, v2, &LPP::add_helper);
    }
    return v1;
}

std::vector<float>& operator+=(std::vector<float>& v1, const float c)
{
    if (v1.size() < LPP::CONSTANTS::VECTOR_PARALLEL_THRESHOLD) {
        for (int i = 0; i < v1.size(); i++) v1[i] += c;
    } else {
        LPP::parallel_comp(v1, c, &LPP::add_helper_scalar);
    }
    return v1;
}

std::vector<float>& operator-=(std::vector<float>& v1, const std::vector<float>& v2)
{
    if (v1.size() != v2.size()) {
        const auto msg = "Subtracting vectors of sizes " + std::to_string(v1.size()) + " and "  + std::to_string(v2.size());
        throw std::invalid_argument(msg);
    }

    if (v1.size() < LPP::CONSTANTS::VECTOR_PARALLEL_THRESHOLD) {
        for (size_t i = 0; i < v1.size(); i++) v1[i] -= v2[i];
    } else {
        LPP::parallel_comp(v1, v2, &LPP::sub_helper);
    }
    return v1;
}

std::vector<float>& operator*=(std::vector<float>& v1, const float c)
{
    if (v1.size() < LPP::CONSTANTS::VECTOR_PARALLEL_THRESHOLD) {
        for (float& d : v1) d *= c;
    } else {
        LPP::parallel_comp(v1, c, &LPP::mult_helper_scalar);
    }
    return v1;
}

std::vector<float>& operator/=(std::vector<float>& v1, const std::vector<float>& v2)
{
    if (v1.size() != v2.size()) {
        const auto msg = "Division (by-entry) of vectors of sizes " + std::to_string(v1.size()) + " and "  + std::to_string(v2.size());
        throw std::invalid_argument(msg);
    }

    // Division by zero is the user's fault
    if (v1.size() < LPP::CONSTANTS::VECTOR_PARALLEL_THRESHOLD) {
        for (size_t i = 0; i < v1.size(); i++) v1[i] /= v2[i];
    } else {
        LPP::parallel_comp(v1, v2, &LPP::div_helper);
    }
    return v1;
}