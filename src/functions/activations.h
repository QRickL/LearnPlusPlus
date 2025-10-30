#ifndef LPP_ACTIVATIONS_H
#define LPP_ACTIVATIONS_H

#include <unordered_map>
#include <memory>
#include <string>

// Activation functions are used by the layer and network classes to perform inference and training
// Available activation functions:
// * Identity
// * ReLU
// * Sigmoid
// * Tanh

namespace LPP {

class Activation {
public:
    virtual double apply_itself(double x) const = 0;
    virtual double apply_derivative(double x) const = 0;
    virtual const std::string& who() const = 0;

    // Activation is abstract
    virtual ~Activation() = 0;
};

class Identity : public Activation {
public:
    double apply_itself(double x) const override;
    double apply_derivative(double x) const override;
    const std::string& who() const override;

    ~Identity() {}
};

class ReLU : public Activation {
public:
    double apply_itself(double x) const override;
    double apply_derivative(double x) const override;
    const std::string& who() const override;

    ~ReLU() {}
};

class Sigmoid : public Activation {
public:
    double apply_itself(double x) const override;
    double apply_derivative(double x) const override;
    const std::string& who() const override;
    
    ~Sigmoid() {}
};

class Tanh : public Activation {
public:
    double apply_itself(double x) const override;
    double apply_derivative(double x) const override;
    const std::string& who() const override;
    
    ~Tanh() {}
};

// Global variables for construction of models by user
const auto IDENTITY = std::make_shared<Identity>();
const auto RELU     = std::make_shared<ReLU>();
const auto SIGMOID  = std::make_shared<Sigmoid>();
const auto TANH     = std::make_shared<Tanh>();

// Global variables for construction of models by file
const std::string STRING_IDENTITY  = "IDENTITY";
const std::string STRING_RELU      = "RELU";
const std::string STRING_SIGMOID   = "SIGMOID";
const std::string STRING_TANH      = "TANH";

const std::unordered_map<std::string, std::shared_ptr<Activation>> choose_activation = {
    {STRING_IDENTITY,   IDENTITY},
    {STRING_RELU,       RELU},
    {STRING_SIGMOID,    SIGMOID},
    {STRING_TANH,       TANH}
};

} // namespace LPP

#endif