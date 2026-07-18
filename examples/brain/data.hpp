#include <opencv2/opencv.hpp>
#include "../../src/LPP.hpp"

const int IMG_DIM = 64;

std::vector<int> g_m_n_p_training = {1400, 1300, 1400, 1400};
std::vector<int> g_m_n_p_validation = {400, 297, 400, 400};

const size_t train_n = 1400 + 1300 + 1400 + 1400;
const size_t val_n = 400 + 297 + 400 + 400;
const size_t network_input_size = IMG_DIM * IMG_DIM;

auto training_X = LPP::Matrix(train_n, network_input_size);
auto training_Y = LPP::Matrix(train_n, 4);
auto validation_X = LPP::Matrix(val_n, network_input_size);
auto validation_Y = LPP::Matrix(val_n, 4);


// Read in image as a 64x64 grayscale image
std::vector<float> load_image(const std::string& path)
{
    cv::Mat image = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if (image.empty())
        throw std::runtime_error("Did not load: " + path);

    cv::resize(image, image, cv::Size(IMG_DIM, IMG_DIM));

    // Convert to float in [0,1]
    image.convertTo(image, CV_32F, 1.0f / 255.0f);

    std::vector<float> input;
    input.reserve(IMG_DIM * IMG_DIM);

    for (int r = 0; r < image.rows; ++r)
    {
        const float* row = image.ptr<float>(r);
        for (int c = 0; c < image.cols; ++c) input.push_back(row[c]);
    }
    return input;
}

void load_Y() {
    for (size_t i = 0; i < train_n; i++)
    {
        if (0 <= i && i < 1400) {
            training_Y.set_row(i, {1.f,0.f,0.f,0.f});
        }
        else if (1400 <= i && i < 2700) {
            training_Y.set_row(i, {0.f,1.f,0.f,0.f});
        }
        else if (2700 <= i && i < 4100) {
            training_Y.set_row(i, {0.f,0.f,1.f,0.f});
        }
        else if (4100 <= i && i < 5500) {
            training_Y.set_row(i, {0.f,0.f,0.f,1.f});
        }
    }

    for (size_t i = 0; i < val_n; i++)
    {
        if (0 <= i && i < 400) {
            validation_Y.set_row(i, {1.f,0.f,0.f,0.f});
        }
        else if (400 <= i && i < 697) {
            validation_Y.set_row(i, {0.f,1.f,0.f,0.f});
        }
        else if (697 <= i && i < 1097) {
            validation_Y.set_row(i, {0.f,0.f,1.f,0.f});
        }
        else if (1097 <= i && i < 1497) {
            validation_Y.set_row(i, {0.f,0.f,0.f,1.f});
        }
    }
}

void load_X() {
    size_t overall_train = 0;
    size_t overall_val = 0;
    std::string path_train = "/LOCATION/OF/TRAINING/DATA/";
    std::string path_val = "/LOCATION/OF/VALIDATION/DATA/";

    path_train = "/Users/rick/Desktop/Coding/LearnPlusPlus/examples/brain/archive/Training/";
    path_val = "/Users/rick/Desktop/Coding/LearnPlusPlus/examples/brain/archive/Testing/";

    std::vector<std::pair<std::string, size_t>> p_train = {
        {"glioma/Tr-gl_", 1400},
        {"meningioma/Tr-me_", 1300},
        {"notumor/Tr-no_", 1400},
        {"pituitary/Tr-pi_", 1400}
    };

    std::vector<std::pair<std::string, size_t>> p_val = {
        {"glioma/Te-gl_", 400},
        {"meningioma/Te-me_", 297},
        {"notumor/Te-no_", 400},
        {"pituitary/Te-pi_", 400}
    };

    for (const auto& p : p_train) {
        for (size_t class_idx = 0; class_idx < p.second; class_idx++) {
            auto f = path_train + p.first + std::to_string(1+class_idx) + ".jpg";
            training_X.set_row(
                overall_train, load_image(f));
            overall_train++;
        }
    }

    for (const auto& p : p_val) {
        for (size_t class_idx = 0; class_idx < p.second; class_idx++) {
            auto f = path_val + p.first + std::to_string(1+class_idx) + ".jpg";
            validation_X.set_row(
                overall_val, load_image(f));
            overall_val++;
        }
    }
}