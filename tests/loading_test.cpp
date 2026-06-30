#include "../src/LPP.hpp"

int main() {

    LPP::Network my_model("saving_test.txt");

    const std::vector<float> classification = my_model.inference(
        {5, 13, -1, 1, 1.4, 0.23, 0, 0, 1, 0}
    );

    LPP::print_object(classification);
}