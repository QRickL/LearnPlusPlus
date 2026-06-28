#include "lpp_print.h"

template <typename number>
void LPP::print_object(const Vect<number>& v, std::ostream& os)
{
    std::cout << '{';
    for (size_t i = 0; i < v.size(); i++)
    std::cout << '}' << std::endl;
}

template <typename number>
void LPP::print_object(const Matrix<number>& m, std::ostream& os) {m.print_entries(os);}