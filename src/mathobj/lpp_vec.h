# pragma once

#include <cstddef> 
#include <iostream>

namespace LPP {

template <typename number>
class Vect {
    const size_t size__;    // Size. Cannot change
    number* data__;         // Data

public:
    Vect(size_t size);
    Vect(size_t size, number initVal);
    Vect(const Vect<number>& other);
    ~Vect();

    // Random access operator. For writing. User's fault for calling out of bounds.
    number& operator[](size_t index) {return data__[index];}

    // Random access operator. For reading. User's fault for calling out of bounds.
    const number& operator[](size_t) const {return data__[index];}

    // Size of vector
    size_t size() const {return size__;}

    // Prints all elements of the vector
    void printElements(std::ostream& os = std::cout) const;
};


}