#ifndef LPP_ACTIVATIONS_HPP
#define LPP_ACTIVATIONS_HPP

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
namespace activations {

class Activation {
public:
    virtual float apply_activation(float x) const = 0;
    virtual float apply_derivative(float x) const = 0;
    virtual const std::string& who() const = 0;

    // Activation is abstract
    virtual ~Activation() = 0;
};

class Identity : public Activation {
public:
    float apply_activation(float x) const override;
    float apply_derivative(float x) const override;
    const std::string& who() const override;

    ~Identity() {}
};

class ReLU : public Activation {
public:
    float apply_activation(float x) const override;
    float apply_derivative(float x) const override;
    const std::string& who() const override;

    ~ReLU() {}
};

class Sigmoid : public Activation {
public:
    float apply_activation(float x) const override;
    float apply_derivative(float x) const override;
    const std::string& who() const override;
    
    ~Sigmoid() {}
};

class Tanh : public Activation {
public:
    float apply_activation(float x) const override;
    float apply_derivative(float x) const override;
    const std::string& who() const override;
    
    ~Tanh() {}
};

// Global variables for construction of models by user

const auto identity = std::make_shared<Identity>();
const auto relu     = std::make_shared<ReLU>();
const auto sigmod  = std::make_shared<Sigmoid>();
const auto tanh     = std::make_shared<Tanh>();

// Global variables for construction of models by file
const std::string STRING_IDENTITY_  = "IDENTITY";
const std::string STRING_RELU_      = "RELU";
const std::string STRING_SIGMOID_   = "SIGMOID";
const std::string STRING_TANH_      = "TANH";

const std::unordered_map<std::string, std::shared_ptr<Activation>> choose_activation = {
    {STRING_IDENTITY_,   activations::identity},
    {STRING_RELU_,       activations::relu},
    {STRING_SIGMOID_,    activations::sigmod},
    {STRING_TANH_,       activations::tanh}
};

} // namespace activations
} // namespace LPP

#endif