#pragma once

#include <unordered_map>
#include <memory>
#include <string>

/*

Activation functions are used by the layer and network classes to perform inference and training
Available activation functions:
* Identity
* ReLU
* Sigmoid
* Tanh

Overides for sigmoid or tanh are required if using custom number types

*/

// TODO: who() should now also report the data type

namespace LPP {

template <typename number>
class Activation {
public:
    virtual number applyActivation(number x) const = 0;
    virtual number applyDerivative(number x) const = 0;
    virtual const std::string& who() const = 0;

    // Activation is abstract
    virtual ~Activation() = 0;
};

template <typename number>
class Identity : public Activation<number> {
public:
    number applyActivation(number x) const override;
    number applyDerivative(number x) const override;
    const std::string& who() const override;

    ~Identity() {}
};

template <typename number>
class ReLU : public Activation<number> {
public:
    number applyActivation(number x) const override;
    number applyDerivative(number x) const override;
    const std::string& who() const override;

    ~ReLU() {}
};

template <typename number>
class Sigmoid : public Activation<number> {
public:
    number applyActivation(number x) const override;
    number applyDerivative(number x) const override;
    const std::string& who() const override;
    
    ~Sigmoid() {}
};

template <typename number>
class Tanh : public Activation<number> {
public:
    number applyActivation(number x) const override;
    number applyDerivative(number x) const override;
    const std::string& who() const override;
    
    ~Tanh() {}
};

// // Global variables for construction of models by user
// const auto IDENTITY = std::make_shared<Identity>();
// const auto RELU     = std::make_shared<ReLU>();
// const auto SIGMOID  = std::make_shared<Sigmoid>();
// const auto TANH     = std::make_shared<Tanh>();

// Global variables for construction of models by file
const std::string STRING_IDENTITY  = "IDENTITY";
const std::string STRING_RELU      = "RELU";
const std::string STRING_SIGMOID   = "SIGMOID";
const std::string STRING_TANH      = "TANH";

template <typename number>
const std::unordered_map<std::string, std::shared_ptr<Activation<number>>> chooseActivation = {
    {STRING_IDENTITY,   std::make_shared< Identity<number> >},
    {STRING_RELU,       std::make_shared< ReLU<number> >},
    {STRING_SIGMOID,    std::make_shared< Sigmoid<number> >},
    {STRING_TANH,       std::make_shared< Tanh<number> >}
};

} // namespace LPP