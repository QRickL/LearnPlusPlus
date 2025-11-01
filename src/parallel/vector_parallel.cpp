#include "vector_parallel.h"
#include <thread>

void LPP::add_helper(std::vector<double>& v1, const std::vector<double>& v2, const size_t start, const size_t end)
{
    for (size_t i = start; i < end; i++) {
        v1[i] += v2[i];
    }
}

void LPP::sub_helper(std::vector<double>& v1, const std::vector<double>& v2, const size_t start, const size_t end)
{
    for (size_t i = start; i < end; i++) {
        v1[i] -= v2[i];
    }
}

void LPP::div_helper(std::vector<double>& v1, const std::vector<double>& v2, const size_t start, const size_t end)
{
    for (size_t i = start; i < end; i++) {
        v1[i] /= v2[i];
    }
}

void LPP::add_helper_scalar(std::vector<double>& v1, const double c, const size_t start, const size_t end)
{
    for (size_t i = start; i < end; i++) {
        v1[i] += c;
    }
}

void LPP::mult_helper_scalar(std::vector<double>& v1, const double c, const size_t start, const size_t end)
{
    for (size_t i = start; i < end; i++) {
        v1[i] *= c;
    }
}

void LPP::parallel_comp(
    std::vector<double>& v1,
    const std::vector<double>& v2,
    void (*helper_func)(
        std::vector<double>& v1,
        const std::vector<double>& v2,
        const size_t start,
        const size_t end
    )
)
{
    // use_threads = 4
    // size = 17
    // 0 1 2 3 | 4 5 6 7 | 9 10 11 12 | 13 14 15 16 | 17

    std::vector<std::thread> blocks;
    const size_t block_size = v1.size() / LPP::USE_THREADS;

    for (size_t th = 0; th < LPP::USE_THREADS; th++) {
        const size_t start = th * block_size;
        const size_t end   = std::min( (th + 1) * block_size, v1.size() );

        blocks.emplace_back(helper_func, std::ref(v1), std::cref(v2), start, end);
    }
    for (auto& block : blocks) block.join();
}

void LPP::parallel_comp(
    std::vector<double>& v1,
    const double c,
    void (*helper_func)(
        std::vector<double>& v1,
        const double c,
        const size_t start,
        const size_t end
    )
)
{
    std::vector<std::thread> blocks;
    const size_t block_size = v1.size() / LPP::USE_THREADS;

    for (size_t th = 0; th < LPP::USE_THREADS; th++) {
        const size_t start = th * block_size;
        const size_t end   = std::min( (th + 1) * block_size, v1.size() );

        blocks.emplace_back(helper_func, std::ref(v1), c, start, end);
    }
    for (auto& block : blocks) block.join();
}