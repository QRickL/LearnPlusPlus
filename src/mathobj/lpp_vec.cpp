#include "lpp_vec.h"

template <typename number>
LPP::Vect<number>::Vect(size_t size) : size__{size}
{
    data__ = new number[size__];
}

template <typename number>
LPP::Vect<number>::Vect(size_t size, number initVal) : size__{size}
{
    data__ = new number[size__];
    for (size_t i = 0; i < size__; i++) data__[i] = initVal;
}

template <typename number>
LPP::Vect<number>::Vect(const Vect& other) : size__{other.size()}
{
    data__ = new number[size__];
    for (size_t i = 0; i < size__; i++) data__[i] = other[i];
}

template <typename number>
LPP::Vect<number>::~Vect() {delete[] data__;}

template <typename number>
void LPP::Vect<number>::printElements(std::ostream& os) const
{
    os << "{ ";
    for (size_t i = 0; i < v.size(); i++) os << data__[i] << ' '
    os << '}' << std::endl;
}