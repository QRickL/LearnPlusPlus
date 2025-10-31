#include "vector_overloads.h"

// Later: see if we stil need some of the vectors, that way we can rewrite the input and save copying

// TODO: parallelize
std::vector<double> operator+(const std::vector<double>& v1, const std::vector<double>& v2) 
{
    if (v1.size() != v2.size()) {
        const auto msg = "Adding vectors of sizes " + std::to_string(v1.size()) + " and "  + std::to_string(v2.size());
        throw std::invalid_argument(msg);
    }

    std::vector<double> res(v1.size());
    for (size_t i = 0; i < v1.size(); i++) {
        res[i] = v1[i] + v2[i];
    }
    return res;
}

// TODO: parallelize
std::vector<double> operator-(const std::vector<double>& v1, const std::vector<double>& v2)
{
    if (v1.size() != v2.size()) {
        const auto msg = "Subtracting vectors of sizes " + std::to_string(v1.size()) + " and "  + std::to_string(v2.size());
        throw std::invalid_argument(msg);
    }

    std::vector<double> res(v1.size());
    for (size_t i = 0; i < v1.size(); i++) {
        res[i] = v1[i] - v2[i];
    }
    return res;
}

// TODO: parallelize
double operator*(const std::vector<double>& v1, const std::vector<double>& v2)
{
    if (v1.size() != v2.size()) {
        const auto msg = "Dotting vectors of sizes " + std::to_string(v1.size()) + " and "  + std::to_string(v2.size());
        throw std::invalid_argument(msg);
    }

    double res = 0.0;
    for (size_t i = 0; i < v1.size(); i++) {
        res += v1[i]*v2[i];
    }
    return res;
}

// TODO: parallelize
std::vector<double> operator+(const std::vector<double>& v1, const double c)
{
    std::vector<double> res(v1.size());
    for (int i = 0; i < v1.size(); i++) {
        res[i] = c + v1[i];
    }
    return res;
}

std::vector<double> operator+(const double c, const std::vector<double>& v2) {
    return v2 + c;
}

// TODO: parallelize
std::vector<double> operator*(const std::vector<double>& v1, const double c)
{
    std::vector<double> res(v1.size());
    for (int i = 0; i < v1.size(); i++) {
        res[i] = c * v1[i];
    }
    return res;
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

    std::vector<double> res(v1.size());
    for (size_t i = 0; i < v1.size(); i++) {
        res[i] = v1[i] / v2[i];
    }
    return res;
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