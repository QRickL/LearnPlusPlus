#ifndef MODEL_H
#define MODEL_H

#include "misc.h"

const double DEFAULT_LEARNING_RATE = 0.0000005;

class Model {

    protected:
        double learningRate;
        bool compiled;

    public:
        Model();
        virtual ~Model() = 0;

        // TO DO: compile()
        virtual void compile();
        // fit() should only run after compile()
        virtual void fit(const std::vector<std::vector<double>> &trainingData, const std::vector<double> &labels, const std::size_t epochs) = 0;
        // predict() should only run after fit()
        virtual std::vector<double> predict(const std::vector<std::vector<double>> &inputs) = 0;
};

#endif
