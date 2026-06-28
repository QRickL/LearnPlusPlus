#include "lpp_vec.h"

template <typename number>
LPP::Vect<number>::Vect(size_t size) : size__{size}
{
    data__ = new number[size];
}

template <typename number>
LPP::Vect<number>::Vect(size_t size, number initVal) : size__{size}
{
    data__ = new number[size];
    for (size_t i = 0; i < size__) data__[i] = initVal;
}

template <typename number>
LPP::Vect<number>::~Vect() {delete data__;}