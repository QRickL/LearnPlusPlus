#include "classify.hpp"
#include "../checking/check.hpp"

size_t LPP::classify::to_label_idx(std::span<const float> x)
{
    size_t ans = -1;
    float max_prob = 0.f;

    for (size_t label = 0; label < x.size(); label++)
    {
        float p = x[label];
        __lpp_check__(p >= 0.f, "to_label_idx - predicted probability less than 0");
        __lpp_check__(p <= 1.f, "to_label_idx - predicted probability greater than 0");

        if (p > max_prob) {
            ans = label;
            max_prob = p;
        }
    }
    return ans;
}

float LPP::classify::compute_accuracy(const LPP::Matrix& actual, const LPP::Matrix& estimated)
{
    __lpp_check__(same_dims(actual, estimated),
        "compute_accuracy - actual and estimated responses have different dimensions");

    const size_t n = actual.rows();
    size_t correct = 0;
    for (size_t i = 0; i < n; i++)
    {
        if (to_label_idx(actual[i]) == to_label_idx(estimated[i])) correct++;
    }
    return ((float) correct) / ((float) n);
}
