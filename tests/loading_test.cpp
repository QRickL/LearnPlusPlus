#include "../src/LPP.h"

int main() {

    LPP::Network my_model("saving_test.txt");

    const std::vector<double> classification = my_model.inference(
        {5, 13, -1, 1, 1.4, 0.23, 0, 0, 1, 0}
    );

    LPP::print_object(classification);
}