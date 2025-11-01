#include "../src/LPP.h"

int main() {

    LPP::Network my_model(
        10,         // Input size of 10
        {
            {8, LPP::RELU},
            {6, LPP::RELU},
            {3, LPP::RELU},
            {1, LPP::SIGMOID}
        },
        LPP::STANDARD_NORMAL
    );

    my_model.save_model("saving_test.txt");
    
}