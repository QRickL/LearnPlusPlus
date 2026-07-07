#ifndef LPP_ACTIVATIONS_HPP
#define LPP_ACTIVATIONS_HPP

#include <unordered_map>
#include <memory>
#include <string>
#include <vector>

// Activation functions are used by the layer and network classes to perform inference and training
// Available activation functions:
// * Identity
// * ReLU
// * Sigmoid
// * Tanh
// * Softmax

namespace LPP {
namespace activations {

class Activation {
    /*
    We say an activation function is non-interdependent if the
    i-th element of the output ONLY relies on the i-th element of the input
    */
    virtual bool elements_non_interdependent_() const = 0;
    virtual float apply_activation_(float x) const = 0;

public:
    virtual void  apply_activation(std::vector<float>& v) const; // vector is modified in place
    virtual float apply_derivative(float x) const = 0;
    virtual const std::string& who() const = 0;

    // Activation is abstract
    virtual ~Activation() = 0;
};

class Identity : public Activation {
    virtual bool elements_non_interdependent_() const {return true;}
    float apply_activation_(float x) const override;

public:
    float apply_derivative(float x) const override;
    const std::string& who() const override;

    ~Identity() {}
};

class ReLU : public Activation {
    virtual bool elements_non_interdependent_() const {return true;}
    float apply_activation_(float x) const override;

public:
    float apply_derivative(float x) const override;
    const std::string& who() const override;

    ~ReLU() {}
};

class Sigmoid : public Activation {
    virtual bool elements_non_interdependent_() const {return true;}
    float apply_activation_(float x) const override;

public:
    float apply_derivative(float x) const override;
    const std::string& who() const override;
    
    ~Sigmoid() {}
};

class Tanh : public Activation {
    virtual bool elements_non_interdependent_() const {return true;}
    float apply_activation_(float x) const override;

public:
    float apply_derivative(float x) const override;
    const std::string& who() const override;
    
    ~Tanh() {}
};

class Softmax : public Activation {
    virtual bool elements_non_interdependent_() const {return false;}
    float apply_activation_(float x) const override;

public:
    void apply_activation(std::vector<float>& v) const override;
    float apply_derivative(float x) const override;
    const std::string& who() const override;
    
    ~Softmax() {}
};

// Global variables for construction of models by user
// Inline variables have a new meaning in C++17. No multiple definition errors

inline const auto identity = Identity();
inline const auto relu     = ReLU();
inline const auto sigmoid  = Sigmoid();
inline const auto tanh     = Tanh();
inline const auto softmax  = Softmax();

// Global variables for construction of models by file
inline const std::string STRING_IDENTITY_  = "IDENTITY";
inline const std::string STRING_RELU_      = "RELU";
inline const std::string STRING_SIGMOID_   = "SIGMOID";
inline const std::string STRING_TANH_      = "TANH";
inline const std::string STRING_SOFTMAX_   = "SOFTMAX";

inline const std::unordered_map<std::string, const Activation*> choose_activation = {
    {STRING_IDENTITY_, &identity},
    {STRING_RELU_,     &relu},
    {STRING_SIGMOID_,  &sigmoid},
    {STRING_TANH_,     &tanh},
    {STRING_SOFTMAX_,  &softmax}
};

} // namespace activations
} // namespace LPP

#endif