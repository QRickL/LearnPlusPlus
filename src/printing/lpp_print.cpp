#include "lpp_print.h"

template <typename number>
void LPP::printObject(const Vect<number>& v, std::ostream& os) {v.printElements(os);}

template <typename number>
void LPP::printObject(const Matrix<number>& m, std::ostream& os) {m.printEntries(os);}

template <typename number>
void LPP::printObject(const Layer<number>& l, std::ostream& os) {l.printLayer(os);}