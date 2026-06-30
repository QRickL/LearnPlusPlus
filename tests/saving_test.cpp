#include "../src/LPP.h"

int main() {

    LPP::Network my_model(
        10,         // Input size of 10
        {
            {8, LPP::activations::RELU},
            {6, LPP::activations::RELU},
            {3, LPP::activations::RELU},
            {1, LPP::activations::SIGMOID}
        },
        LPP::standard_normal_distn
    );

    my_model.save_model("saving_test.txt");
    
}