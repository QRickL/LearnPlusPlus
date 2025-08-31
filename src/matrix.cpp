#include "matrix.h"

const unsigned int MAX_THREADS = std::thread::hardware_concurrency();       // How many threads are available
const unsigned int USE_THREADS = std::max(1u, std::min(4u, MAX_THREADS));   // Don't use up all threads
const size_t PARALLEL_THRESHOLD = 10000;                                    // Will not parallelize for small vectors/matrices

void add_vectors_helper(std::vector<double>& result, const std::vector<double>& v1, const std::vector<double>& v2, size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
        result[i] = v1[i] + v2[i];
    }
}

std::vector<double> add_vectors(const std::vector<double>& v1, const std::vector<double>& v2) {
    if (v1.size() != v2.size()) throw std::runtime_error("Adding vectors of different sizes");
    
    std::vector<double> result(v1.size());
    // ***
    // This step uses a lot of time... see if we can reuse something
    // ***

    if (result.size() < PARALLEL_THRESHOLD) {
        add_vectors_helper(result, v1, v2, 0, result.size());
        return result;
    }

    // Sizing
    std::vector<std::thread> threads;
    threads.reserve(USE_THREADS);
    const size_t chunk_size = result.size() / USE_THREADS;

    // Vectorized operations
    for (size_t c = 0; c < USE_THREADS; c++) {
        const size_t start = chunk_size * c;
        const size_t end = (c == USE_THREADS - 1) ? result.size() : start + chunk_size;
        threads.emplace_back(add_vectors_helper, std::ref(result), std::ref(v1), std::ref(v2), start, end);
    }
    for (auto& t : threads) t.join();

    return result;
}

void dot_prod_helper(double& result, const std::vector<double>& v1, const std::vector<double>& v2, size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
        result += v1[i] * v2[i];
    }
}

double dot_prod(const std::vector<double>& v1, const std::vector<double>& v2) {
    if (v1.size() != v2.size()) throw std::runtime_error("Adding vectors of different sizes");
    double result = 0.0;

    if (v1.size() < PARALLEL_THRESHOLD) {
        dot_prod_helper(result, v1, v2, 0, v1.size());
        return result;
    }

    // Sizing
    std::vector<std::thread> threads;
    threads.reserve(USE_THREADS);
    const size_t chunk_size = v1.size() / USE_THREADS;

    // Vectorized operations
    for (size_t c = 0; c < USE_THREADS; c++) {
        const size_t start = chunk_size * c;
        const size_t end = (c == USE_THREADS - 1) ? v1.size() : start + chunk_size;
        threads.emplace_back(dot_prod_helper, std::ref(result), std::ref(v1), std::ref(v2), start, end);
    }
    for (auto& t : threads) t.join();

    return result;
}

Matrix::Matrix(const std::vector<std::vector<double>>& m) : entry{m.size()} {
    if (m.empty()) {
        throw std::runtime_error("Bad vector-vector to matrix initialization (empty)");
    }
    const size_t cols = m[0].size();

    for (size_t i = 0; i < m.size(); i++) {
        if (m[i].size() != cols) {
            throw std::runtime_error("Bad vector-vector to matrix initialization (dimensions)");
        }
        entry[i] = std::move(m[i]);
    }
}

Matrix::Matrix(const size_t row, const size_t column) : entry{row} {
    for (size_t r = 0; r < row; r++) {
        entry[r].resize(column);
    }
}

size_t Matrix::getRows() const {
    return entry.size();
}

size_t Matrix::getCols() const {
    if (entry.empty()) {
        return 0;
    }
    return entry[0].size();
}

std::vector<double>& Matrix::operator[](size_t row) {
    return entry[row];
}

void Matrix::multiplication_helper(std::vector<double>& result, const std::vector<double>& x, size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
        result[i] = dot_prod(x, entry[i]);
    }
}

std::vector<double> Matrix::operator*(std::vector<double>& x) {
    if (getCols() != x.size()) {
        throw std::runtime_error("Bad matrix vector multiplication (dimensions)");
    }

    std::vector<double> result(entry.size());

    if (result.size() < PARALLEL_THRESHOLD) {
        multiplication_helper(result, x, 0, result.size());
        return result;
    }

    // Vectorized operations
    std::vector<std::thread> threads;
    threads.reserve(USE_THREADS);
    const size_t chunk_size = result.size() / USE_THREADS;

    for (size_t c = 0; c < USE_THREADS; c++) {
        const size_t start = chunk_size * c;
        const size_t end = (c == USE_THREADS - 1) ? result.size() : start + chunk_size;

        threads.emplace_back(&Matrix::multiplication_helper, this, std::ref(result), std::ref(x), start, end);
    }
    for (auto& t : threads) t.join();

    return result;
}