#ifndef MODEL_H
#define MODEL_H

#include "misc.h"

const double DEFAULT_LEARNING_RATE = 0.0000005;

class Model {

    protected:
        double learningRate;

    public:
        Model();
        virtual ~Model() = 0;

        virtual void fit(const std::vector<std::vector<double>> &trainingData, const std::vector<double> &labels, const std::size_t epochs) = 0;
        virtual void compile();
        // TO DO: compile()
        virtual std::vector<double> predict(const std::vector<std::vector<double>> &inputs) = 0;
};

#endif
