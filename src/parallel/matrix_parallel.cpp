#include "matrix_parallel.h"
#include <thread>

#include <iostream>

void LPP::matrix_init_helper(std::vector<std::vector<double>>& entries, const size_t cols, const size_t start_row, const size_t end_row, const double c)
{
    for (size_t r = start_row; r < end_row; r++) {
        entries[r] = std::vector<double>(cols, c);
    }
}

void LPP::matrix_parallel_init(std::vector<std::vector<double>>& entries, const size_t rows, const size_t cols, const double c)
{
    std::vector<std::thread> blocks;
    const size_t block_size = rows / LPP::USE_THREADS;

    for (size_t r = 0; r <= LPP::USE_THREADS; r++) {
        const size_t start  = r * block_size;
        const size_t end    = std::min( (r + 1) * block_size, rows );

        blocks.emplace_back(LPP::matrix_init_helper, std::ref(entries), cols, start, end, c);
    }
    for (auto& block : blocks) block.join();
}