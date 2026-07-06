#include "../src/LPP.hpp"

int main() {

    LPP::Network my_model(
        10,
        {
            {8, &LPP::activations::relu},
            {6, &LPP::activations::relu},
            {3, &LPP::activations::relu},
            {1, &LPP::activations::sigmoid}
        },
        &LPP::distribution::standard_normal
    );

    my_model.save_model("saving_test.txt");
    
}