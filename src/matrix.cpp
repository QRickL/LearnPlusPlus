#include "matrix.h"
//#include "parallel/matrix_parallel.h"
#include <exception>
#include <iostream>
#include <random>

LPP::Matrix::Matrix(const std::vector<std::vector<float>>& m) {entries = m;}

LPP::Matrix::Matrix(std::vector<std::vector<float>>& m) {entries = std::move(m);}

LPP::Matrix::Matrix(size_t rows, size_t cols): entries{rows}
{
    for (size_t r = 0; r < rows; r++) {
        entries[r] = std::vector<float>(cols, 0.0);
    }
    // if (rows < LPP::CONSTANTS::MATRIX_PARALLEL_THRESHOLD) {
    //     for (size_t r = 0; r < rows; r++) {
    //         entries[r] = std::vector<float>(cols, 0.0);
    //     }
    // } else {
    //     LPP::matrix_parallel_init(entries, rows, cols, 0.0);
    // }
}

LPP::Matrix::Matrix(size_t rows, size_t cols, float c): entries{rows}
{
    for (size_t r = 0; r < rows; r++) {
        entries[r] = std::vector<float>(cols, c);
    }
    // if (rows < LPP::CONSTANTS::MATRIX_PARALLEL_THRESHOLD) {
    //     for (size_t r = 0; r < rows; r++) {
    //         entries[r] = std::vector<float>(cols, c);
    //     }
    // } else {
    //     LPP::matrix_parallel_init(entries, rows, cols, c);
    // }
}

LPP::Matrix::Matrix(size_t rows, size_t cols, const std::shared_ptr<LPP::ProbabilityDistribution>& pd): entries{rows}
{
    for (size_t r = 0; r < rows; r++) {
        entries[r].resize(cols);
        for (size_t c = 0; c < cols; c++) {
            entries[r][c] = pd->sample();
        }
    }
}

size_t LPP::Matrix::rows() const
{
    return entries.size();
}

size_t LPP::Matrix::cols() const
{
    if (entries.empty()) return 0;
    return entries[0].size();
}

void LPP::Matrix::print_entries() const
{
    std::cout << "{\n";
    for (size_t i = 0; i < entries.size(); i++) {
        print_object(entries[i]);
    }
    std::cout << '}' << std::endl;
}

float LPP::Matrix::get(size_t i, size_t j) const
{
    if (i >= entries.size() || j >= entries[0].size()) {
        throw std::invalid_argument("Vector entry extraction out of range");
    }
    return entries[i][j];
}

void LPP::Matrix::set(size_t i, size_t j, float s)
{
    if (i >= entries.size() || j >= entries[0].size()) {
        throw std::invalid_argument("Vector entry setting out of range");
    }
    entries[i][j] = s;
}

const std::vector<float>& LPP::Matrix::operator[](size_t i) const {return entries[i];}

std::vector<float>& LPP::Matrix::operator[](size_t i) {return entries[i];}

std::vector<float> LPP::Matrix::operator*(const std::vector<float>& v) const
{
    if (entries.empty()) {
        const auto msg = "Attempting to multiply by empty matrix";
        throw std::invalid_argument(msg);
    }
    if (entries[0].size() != v.size()) {
        const auto msg = "Attempting to multiply " + std::to_string(entries.size()) + 'x' + std::to_string(entries[0].size()) + " by a 1x" + std::to_string(v.size()) + " vector";
        throw std::invalid_argument(msg);
    }

    std::vector<float> res(entries.size());
    for (size_t i = 0; i < res.size(); i++) {
        res[i] = entries[i] * v;
    }
    return res;
}

void LPP::matrix_sub_helper(Matrix& m1, const LPP::Matrix& m2, size_t start, size_t end) {
    for (size_t r = start; r < end; r++) {
        m1[r] -= m2[r];
    }
}

LPP::Matrix& LPP::Matrix::operator-=(const Matrix& m)
{
    if (!same_dims((*this), m)) {
        const auto msg = "Attempting to subtract matrices of different sizes";
        throw std::invalid_argument(msg);
    }

    for (size_t i = 0; i < rows(); i++) {
        entries[i] -= m.entries[i];
    }
    return *this;
    
    // if (this->rows() < LPP::CONSTANTS::MATRIX_PARALLEL_THRESHOLD) {
    //     for (size_t i = 0; i < rows(); i++) {
    //         entries[i] -= m.entries[i];
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

void LPP::matrix_mult_helper(Matrix& m1, size_t start, size_t end, float c)
{
    for (size_t r = start; r < end; r++) {
        m1[r] *= c;
    }
}

LPP::Matrix& LPP::Matrix::operator*=(float c)
{
    for (size_t i = 0; i < entries.size(); i++) {
        entries[i] *= c;
    }
    return *this;

    // if (this->rows() < LPP::CONSTANTS::MATRIX_PARALLEL_THRESHOLD) {
    //     for (size_t i = 0; i < entries.size(); i++) {
    //         entries[i] *= c;
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

void LPP::print_object(const std::vector<float>& v)
{
    std::cout << '{';
    for (float d : v) {
        std::cout << d << ", ";
    }
    std::cout << '}' << std::endl;
}

void LPP::print_object(const Matrix& m) {m.print_entries();}