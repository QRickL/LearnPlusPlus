#ifndef LINEARREGRESSION_H
#define LINEARREGRESSION_H

#include "Model.h"

class LinearRegression : public Model {
    std::size_t numFeatures;
    double bias;
    std::vector<double> coefficients;

    // Used for quick dot products
    inline double singleton(const std::vector<double> &features);

    public:
        LinearRegression();
        
        // TO DO: compile()
        // void compile() override;
        void fit(const std::vector<std::vector<double>> &trainingData, const std::vector<double> &labels, const std::size_t epochs) override;
        std::vector<double> predict(const std::vector<std::vector<double>> &inputs) override;

        double getBias();
        std::vector<double> getCoefficients();
};

#endif
