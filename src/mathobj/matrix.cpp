#include "matrix.hpp"
#include "../checking/check.hpp"
//#include "parallel/matrix_parallel.h"
#include <exception>
#include <iostream>
#include <random>

LPP::Matrix::Matrix() :
    rows_{0},
    cols_{0},
    entries_{}
{}

LPP::Matrix::Matrix(const std::vector<std::vector<float>>& m) :
    rows_{m.size()},
    cols_{m[0].size()},
    entries_(rows_ * cols_)
{
    for (size_t i = 0; i < rows_; i++)
    {
        for (size_t j = 0; j < cols_; j++)
        {
            entries_[i * cols_ + j] = m[i][j];
        }
    }
}

LPP::Matrix::Matrix(std::vector<std::vector<float>>& m) :
    rows_{m.size()},
    cols_{m[0].size()},
    entries_(rows_ * cols_)
{
    for (size_t i = 0; i < rows_; i++)
    {
        for (size_t j = 0; j < cols_; j++)
        {
            entries_[i * cols_ + j] = m[i][j];
        }
    }
}

LPP::Matrix::Matrix(size_t rows, size_t cols) :
    rows_{rows},
    cols_{cols},
    entries_(rows_ * cols_)
{
    for (size_t i = 0; i < rows_; i++)
    {
        for (size_t j = 0; j < cols_; j++)
        {
            entries_[i * cols_ + j] = 0.f;
        }
    }
    // for (size_t r = 0; r < rows; r++) {
    //     entries_[r] = std::vector<float>(cols, 0.f);
    // }
    // if (rows < LPP::CONSTANTS::MATRIX_PARALLEL_THRESHOLD) {
    //     for (size_t r = 0; r < rows; r++) {
    //         entries_[r] = std::vector<float>(cols, 0.0);
    //     }
    // } else {
    //     LPP::matrix_parallel_init(entries_, rows, cols, 0.0);
    // }
}

LPP::Matrix::Matrix(size_t rows, size_t cols, float c) :
    rows_{rows},
    cols_{cols},
    entries_(rows_ * cols_)
{
    for (size_t i = 0; i < rows_; i++)
    {
        for (size_t j = 0; j < cols_; j++)
        {
            entries_[i * cols_ + j] = c;
        }
    }
    // if (rows < LPP::CONSTANTS::MATRIX_PARALLEL_THRESHOLD) {
    //     for (size_t r = 0; r < rows; r++) {
    //         entries_[r] = std::vector<float>(cols, c);
    //     }
    // } else {
    //     LPP::matrix_parallel_init(entries_, rows, cols, c);
    // }
}

LPP::Matrix::Matrix(size_t rows, size_t cols, LPP::distribution::ProbabilityDistribution* pd) : 
    rows_{rows},
    cols_{cols},
    entries_(rows_ * cols_)
{
    __lpp_check__(pd,
        "Matrix::Matrix - probability distribution parameter is nullptr");
    
    for (size_t i = 0; i < rows_; i++)
    {
        for (size_t j = 0; j < cols_; j++)
        {
            entries_[i * cols_ + j] = pd->sample();
        }
    }
}

size_t LPP::Matrix::rows() const
{
    return rows_;
}

size_t LPP::Matrix::cols() const
{
    return cols_;
}

void LPP::Matrix::print_entries(std::ostream& os) const
{
    os << "{\n";
    for (size_t i = 0; i < entries_.size(); i++) {
        print_object(this->operator[](i), os);
    }
    os << '}' << std::endl;
}

float LPP::Matrix::get(size_t i, size_t j) const
{
    __lpp_check__(i < rows_ && j < cols_,
        "Matrix::get - Vector entry extraction out of range");
    
    return entries_[i * cols_ + j];
}

void LPP::Matrix::set(size_t i, size_t j, float s)
{
    __lpp_check__(i < rows_ && j < cols_,
        "Matrix::set - Vector entry setting out of range");
 
    entries_[i * cols_ + j] = s;
}

std::span<const float> LPP::Matrix::operator[](size_t i) const
{
    __lpp_check__(i < rows_,
        "Matrix::operator[] out of bounds");

    return {entries_.data() + i * cols_, cols_};
}

std::span<float> LPP::Matrix::operator[](size_t i)
{
    return {entries_.data() + i * cols_, cols_};
}

std::vector<float> LPP::Matrix::operator*(const std::vector<float>& v) const
{
    __lpp_check__(!entries_.empty(),
        "Matrix::operator* - Attempting to multiply by empty matrix");
    __lpp_check__(cols_ == v.size(),
        "Matrix::operator* - Attempting to multiply " + std::to_string(rows_) + 'x' + std::to_string(cols_) + " by a 1x" + std::to_string(v.size()) + " vector");
    
    std::vector<float> res(rows_);
    for (size_t i = 0; i < rows_; i++) {
        res[i] = this->operator[](i) * v;
    }
    return res;
}

void LPP::Matrix::mult_add_write(
    const std::vector<float>& to_mult,
    const std::vector<float>& to_add,
    std::vector<float>& to_write
) const {
    // TODO: add checks later

    for (size_t r = 0; r < rows_; r++) {
        to_write[r] = this->operator[](r) * to_mult + to_add[r];
    }
}

// void LPP::matrix_sub_helper(Matrix& m1, const LPP::Matrix& m2, size_t start, size_t end) {
//     for (size_t r = start; r < end; r++) {
//         m1[r] -= m2[r];
//     }
// }

LPP::Matrix& LPP::Matrix::operator-=(const Matrix& m)
{
    __lpp_check__(same_dims(*this, m),
        "Matrix:operator-= - Attempting to subtract matrices of different sizes");

    for (size_t i = 0; i < rows_ * cols_; i++) {
        entries_[i] -= m.entries_[i];
    }
    return *this;
    
    // if (this->rows() < LPP::CONSTANTS::MATRIX_PARALLEL_THRESHOLD) {
    //     for (size_t i = 0; i < rows(); i++) {
    //         entries_[i] -= m.entries_[i];
    //     }
    // } else {
    //     std::vector<std::thread> blocks;
    //     size_t block_size = this->rows() / LPP::CONSTANTS::USE_THREADS;

    //     for (size_t r = 0; r <= LPP::CONSTANTS::USE_THREADS; r++) {
    //         size_t start  = r * block_size;
    //         size_t end    = std::min( (r + 1) * block_size, this->rows() );

    //         blocks.emplace_back(LPP::matrix_sub_helper, std::ref(*this), std::ref(m), start, end);
    //     }
    //     for (auto& block : blocks) block.join();
    // }
    // return *this;
}

// void LPP::matrix_mult_helper(Matrix& m1, size_t start, size_t end, float c)
// {
//     for (size_t r = start; r < end; r++) {
//         m1[r] *= c;
//     }
// }

LPP::Matrix& LPP::Matrix::operator*=(float c)
{
    for (size_t i = 0; i < entries_.size(); i++) {
        entries_[i] *= c;
    }
    return *this;

    // if (this->rows() < LPP::CONSTANTS::MATRIX_PARALLEL_THRESHOLD) {
    //     for (size_t i = 0; i < entries_.size(); i++) {
    //         entries_[i] *= c;
    //     }
    // } else {
    //     std::vector<std::thread> blocks;
    //     size_t block_size = this->rows() / LPP::CONSTANTS::USE_THREADS;

    //     for (size_t r = 0; r <= LPP::CONSTANTS::USE_THREADS; r++) {
    //         size_t start  = r * block_size;
    //         size_t end    = std::min( (r + 1) * block_size, this->rows() );

    //         blocks.emplace_back(LPP::matrix_mult_helper, std::ref(*this), start, end, c);
    //     }
    //     for (auto& block : blocks) block.join();
    // }
    // return *this;
}

bool LPP::same_dims(const Matrix& m1, const Matrix& m2)
{
    return m1.rows() == m2.rows() && m1.cols() == m2.cols();
}

void LPP::print_object(const std::span<const float> v, std::ostream& os)
{
    os << '{';
    for (float d : v) {
        os << d << ", ";
    }
    os << '}' << std::endl;
}

void LPP::print_object(const std::vector<float>& v, std::ostream& os)
{
    os << '{';
    for (float d : v) {
        os << d << ", ";
    }
    os << '}' << std::endl;
}

void LPP::print_object(const Matrix& m, std::ostream& os) {m.print_entries(os);}

float LPP::Matrix::sum_entries_abs() const
{
    float res = 0.f;
    for (size_t i = 0; i < rows_; i++) {
        for (size_t j = 0; j < cols_; j++) {
            res += std::abs(entries_[i * cols_ + j]);
        }
    }
    return res;
}

float LPP::Matrix::sum_entries_sqr() const
{
    float res = 0.f;
    for (size_t i = 0; i < rows_; i++) {
        for (size_t j = 0; j < cols_; j++) {
            float e = entries_[i * cols_ + j];
            res += e*e;
        }
    }
    return res;
}

float LPP::Matrix::sum_entries_elastic(float a) const
{
    float res = 0.f;
    for (size_t i = 0; i < rows_; i++) {
        for (size_t j = 0; j < cols_; j++) {
            float d = entries_[i * cols_ + j];
            res += a * std::abs(d) + (1.f-a) * d * d;
        }
    }
    return res;
}

void LPP::Matrix::set_all(float s)
{
    std::fill(entries_.begin(), entries_.end(), s);
}

void LPP::Matrix::set_row(size_t row, const std::vector<float>& v)
{
    __lpp_check__(row < rows_,
        "Matrix::set_row - setting out of bounds");
    __lpp_check__(cols_ == v.size(),
        "Matrix::set_row - matrix column size not same as vector dimension");

    for (size_t j = 0; j < cols_; j++)
    {
        entries_[row * cols_ + j] = v[j];
    }
}

void LPP::Matrix::normalize()
{
    float mean = 0.0f;
    float pop_var = 0.0f;
    const size_t n = entries_.size();

    // Run over
    for (size_t i = 0; i < n; i++) {
        mean += entries_[i];
        pop_var += entries_[i] * entries_[i];
    }

    mean /= n;
    pop_var = (pop_var / n) - mean * mean;              // Var(x) = E(x^2) - E(x)^2
    pop_var = std::max(pop_var, normalization_epsilon); // Floating point error

    const float sigma = std::sqrtf(pop_var);

    // Standardize
    for (size_t i = 0; i < n; i++) {
        entries_[i] = (entries_[i] - mean) / sigma;
    }
}

void LPP::Matrix::resize_and_set(size_t rows, size_t cols, float s)
{
    rows_ = rows;
    cols_ = cols;
    entries_.resize(rows_ * cols_);
    set_all(s);
}

void LPP::Matrix::resize(size_t rows, size_t cols)
{
    rows_ = rows;
    cols_ = cols;
    entries_.resize(rows_ * cols_);
}