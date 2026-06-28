#include "matrix.h"
// #include "parallel/matrix_parallel.h"
#include <exception>
#include <iostream>
#include <random>

// TODO: parallelization has been removed for now

template <typename number>
LPP::Matrix<number>::Matrix(const std::vector<std::vector<float>>& m) : rows__{m.size()}, cols__{m[0].size()}
{
    entries__ = new number[rows__ * cols__];
    for (size_t i = 0; i < rows__; i++) {
        for (size_t j = 0; j < cols__; j++) {
            entries__[i * rows__ + j] = m[i][j];
        }
    }
}

template <typename number>
LPP::Matrix<number>::Matrix(std::vector<std::vector<float>>& m) : rows__{m.size()}, cols__{m[0].size()}
{
    entries__ = new number[rows__ * cols__];
    for (size_t i = 0; i < rows__; i++) {for (size_t j = 0; j < cols__; j++) {
        entries__[i * rows__ + j] = m[i][j];
    }}
}

template <typename number>
LPP::Matrix<number>::Matrix(size_t rows, size_t cols) : rows__{rows}, cols__{cols}
{
    entries__ = new number[rows__ * cols__];
    for (size_t i = 0; i < rows__; i++) {for (size_t j = 0; j < cols__; j++) {
        entries__[i * rows__ + j] = 0;
    }}

    // if (rows < LPP::CONSTANTS::MATRIX_PARALLEL_THRESHOLD) {
    //     for (size_t r = 0; r < rows; r++) {
    //         entries__[r] = std::vector<double>(cols, 0.0);
    //     }
    // } else {
    //     LPP::matrix_parallel_init(entries__, rows, cols, 0.0);
    // }
}

template <typename number>
LPP::Matrix<number>::Matrix(size_t rows, size_t cols, float initVal) : rows__{rows}, cols__{cols}
{
    entries__ = new number[rows__ * cols__];
    for (size_t i = 0; i < rows__; i++) {for (size_t j = 0; j < cols__; j++) {
        entries__[i * rows__ + j] = initVal;
    }}

    // if (rows < LPP::CONSTANTS::MATRIX_PARALLEL_THRESHOLD) {
    //     for (size_t r = 0; r < rows; r++) {
    //         entries__[r] = std::vector<double>(cols, c);
    //     }
    // } else {
    //     LPP::matrix_parallel_init(entries__, rows, cols, c);
    // }
}

template <typename number>
LPP::Matrix<number>::Matrix(size_t rows, size_t cols, const std::shared_ptr<LPP::ProbabilityDistribution>& pd) : rows__{rows}, cols__{cols}
{
    entries__ = new number[rows__ * cols__];
    for (size_t i = 0; i < rows__; i++) {for (size_t j = 0; j < cols__; j++) {
        entries__[i * rows__ + j] = pd->sample();
    }}
}

template <typename number>
size_t LPP::Matrix<number>::rows() const { return rows__; }

template <typename number>
size_t LPP::Matrix<number>::cols() const { return cols__; }

template <typename number>
void LPP::Matrix<number>::print_entries(std::ostream& os) const
{
    os << "{\n";
    for (size_t i = 0; i < rows__; i++) {
        for (size_t j = 0; j < cols__; j++) {
            os << entries__[i * rows__ + j] << ' ';
        }
        os << '\n';
    }
    os << "}\n" << std::flush;
}

double LPP::Matrix::get(size_t i, size_t j) const
{
    if (i >= entries__.size() || j >= entries__[0].size()) {
        throw std::invalid_argument("Vector entry extraction out of range");
    }
    return entries__[i][j];
}

void LPP::Matrix::set(size_t i, size_t j, double s)
{
    if (i >= entries__.size() || j >= entries__[0].size()) {
        throw std::invalid_argument("Vector entry setting out of range");
    }
    entries__[i][j] = s;
}

const std::vector<double>& LPP::Matrix::operator[](size_t i) const {return entries__[i];}

std::vector<double>& LPP::Matrix::operator[](size_t i) {return entries__[i];}

std::vector<double> LPP::Matrix::operator*(const std::vector<double>& v) const
{
    if (entries__.empty()) {
        const auto msg = "Attempting to multiply by empty matrix";
        throw std::invalid_argument(msg);
    }
    if (entries__[0].size() != v.size()) {
        const auto msg = "Attempting to multiply " + std::to_string(entries__.size()) + 'x' + std::to_string(entries__[0].size()) + " by a 1x" + std::to_string(v.size()) + " vector";
        throw std::invalid_argument(msg);
    }

    std::vector<double> res(entries__.size());
    for (size_t i = 0; i < res.size(); i++) {f
        res[i] = entries__[i] * v;
    }
    return res;
}

// void LPP::matrix_sub_helper(Matrix& m1, const LPP::Matrix& m2, const size_t start, const size_t end) {
//     for (size_t r = start; r < end; r++) {
//         m1[r] -= m2[r];
//     }
// }

LPP::Matrix& LPP::Matrix::operator-=(const Matrix& m)
{
    if (!same_dims((*this), m)) {
        const auto msg = "Attempting to subtract matrices of different sizes";
        throw std::invalid_argument(msg);
    }

    if (this->rows() < LPP::CONSTANTS::MATRIX_PARALLEL_THRESHOLD) {
        for (size_t i = 0; i < rows(); i++) {
            entries__[i] -= m.entries__[i];
        }
    } else {
        // std::vector<std::thread> blocks;
        // const size_t block_size = this->rows() / LPP::CONSTANTS::USE_THREADS;

        // for (size_t r = 0; r <= LPP::CONSTANTS::USE_THREADS; r++) {
        //     const size_t start  = r * block_size;
        //     const size_t end    = std::min( (r + 1) * block_size, this->rows() );

        //     blocks.emplace_back(LPP::matrix_sub_helper, std::ref(*this), std::ref(m), start, end);
        // }
        // for (auto& block : blocks) block.join();
    }
    return *this;
}

// void LPP::matrix_mult_helper(Matrix& m1, const size_t start, const size_t end, const double c)
// {
//     for (size_t r = start; r < end; r++) {
//         m1[r] *= c;
//     }
// }

LPP::Matrix& LPP::Matrix::operator*=(const double c)
{
    if (this->rows() < LPP::CONSTANTS::MATRIX_PARALLEL_THRESHOLD) {
        for (size_t i = 0; i < entries__.size(); i++) {
            entries__[i] *= c;
        }
    } else {
        // std::vector<std::thread> blocks;
        // const size_t block_size = this->rows() / LPP::CONSTANTS::USE_THREADS;

        // for (size_t r = 0; r <= LPP::CONSTANTS::USE_THREADS; r++) {
        //     const size_t start  = r * block_size;
        //     const size_t end    = std::min( (r + 1) * block_size, this->rows() );

        //     blocks.emplace_back(LPP::matrix_mult_helper, std::ref(*this), start, end, c);
        // }
        // for (auto& block : blocks) block.join();
    }
    return *this;
}

template <typename number>
bool LPP::same_dims(const Matrix<number>& m1, const Matrix<number>& m2)
{
    return m1.rows() == m2.rows() && m1.cols() == m2.cols();
}