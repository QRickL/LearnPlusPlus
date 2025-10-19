#ifndef LPP_ACTIVATIONS_H
#define LPP_ACTIVATIONS_H

#include <memory>

// Available activation functions:
// * Identity
// * ReLU
// * Sigmoid
// * Tanh

namespace LPP {

class Activation {
public:
    virtual double apply_itself(double x) const;
    virtual double apply_derivative(double x) const;

    // Activation is abstract
    virtual ~Activation() = 0;
};

class Identity :  public Activation {
public:
    double apply_itself(double x) const override;
    double apply_derivative(double x) const override;

    ~Identity() {}
};

class ReLU : public Activation {
public:
    double apply_itself(double x) const override;
    double apply_derivative(double x) const override;

    ~ReLU() {}
};

class Sigmoid : public Activation {
public:
    double apply_itself(double x) const override;
    double apply_derivative(double x) const override;
    
    ~Sigmoid() {}
};

class Tanh : public Activation {
public:
    double apply_itself(double x) const override;
    double apply_derivative(double x) const override;
    
    ~Tanh() {}
};

// Global variables for user
const auto IDENTITY = std::make_shared<Identity>();
const auto RELU = std::make_shared<ReLU>();
const auto SIGMOID = std::make_shared<Sigmoid>();
const auto TANH = std::make_shared<Tanh>();

} // namespace LPP

#endif