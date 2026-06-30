// #ifndef LPP_VECTOR_PARALLEL_H
// #define LPP_VECTOR_PARALLEL_H

// #include "thread_constants.h"
// #include <vector>

// namespace LPP {

// // Want minimum overhead, avoid templates or function pointers extra operations
// void add_helper(std::vector<double>& v1, const std::vector<double>& v2, const size_t start, const size_t end);
// void sub_helper(std::vector<double>& v1, const std::vector<double>& v2, const size_t start, const size_t end);
// void div_helper(std::vector<double>& v1, const std::vector<double>& v2, const size_t start, const size_t end);
// void add_helper_scalar(std::vector<double>& v1, const double c, const size_t start, const size_t end);
// void mult_helper_scalar(std::vector<double>& v1, const double c, const size_t start, const size_t end);

// void parallel_comp(
//     std::vector<double>& v1,
//     const std::vector<double>& v2,
//     void (*helper_func)(
//         std::vector<double>& v1,
//         const std::vector<double>& v2,
//         const size_t start,
//         const size_t end
//     )
// );

// void parallel_comp(
//     std::vector<double>& v1,
//     const double c,
//     void (*helper_func)(
//         std::vector<double>& v1,
//         const double c,
//         const size_t start,
//         const size_t end
//     )
// );

// } // namespace LPP

// #endif