#include "../src/matrix.h"

int main()
{
    Matrix A({{1, 5, 2, 1},
              {6, 2, 24.555, 62},
              {-2, 5, 1, 1},
              {1, 2, 3, 61},
              {1, 6, 2, 1},
              {4, 5, 2, 2}});
    std::cout << A.getRows() << 'x' << A.getCols() << std::endl;
    std::vector<double> x = {2.56, 5, 5.34, -5};

    auto start = std::chrono::high_resolution_clock::now();

    auto res = A * x;

    auto end = std::chrono::high_resolution_clock::now();

    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms\n";
}