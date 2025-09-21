#include "matrix.h"
#include "mlp.h"
#include "iostream"
#include "activations.cpp"

// try for 5 million total weights

int main()
{

    const long long N = 10000;
    Matrix A(N, N);

    std::vector<double> x(N, 7);

    MultiLayerPerceptron test(7, {{5, sigmoid},
                                  {4, sigmoid},
                                  {6, std::tanh},
                                  {5, sigmoid}});

    test.display_mlp_dimensions();
    test.display_mlp();
    auto resy = test.forward_prop_train({5, 5, 5, 5, 5, 5, 5});

    for (const auto &r : resy)
    {
        std::cout << r << ' ';
    }
    std::cout << '\n';

    // 4 -> 6621ms
    // 1 -> 13797ms
    // 6 -> 7994ms
    // 2 -> 7122ms

    return 0;
}