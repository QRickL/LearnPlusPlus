#include "../src/matrix.h"
#include "../src/matrix.cpp"
#include "chrono"

const int N = 10000;

int main()
{
    std::vector<double> v1(N, 0.5);
    std::vector<double> v2(N, 2.0);

    // std::vector<double> res(N);

    // No
    auto start1 = std::chrono::high_resolution_clock::now();
    // std::vector<double> res(N);
    double res = 0;
    dot_prod_helper(res, v1, v2, 0, v1.size());
    auto end1 = std::chrono::high_resolution_clock::now();
    std::cout << "Time for no para: " << std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1).count() << "ms\n";

    // Yes
    auto start2 = std::chrono::high_resolution_clock::now();
    // auto res2 = add_vectors(v1, v2);
    double res2 = dot_prod(v1, v2);
    auto end2 = std::chrono::high_resolution_clock::now();
    std::cout << "Time using para: " << std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2).count() << "ms\n";
}