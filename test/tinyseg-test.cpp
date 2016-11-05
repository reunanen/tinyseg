#include "../tinyseg.hpp"

int main(int argc, char* argv[])
{
    const std::vector<cv::Scalar> mask_colors = {
        cv::Scalar(127, 127, 127),
        cv::Scalar(255, 0, 0),
        cv::Scalar(0, 255, 0),
        cv::Scalar(0, 0, 255),
    };

    const auto& dataset = tinyseg::load_image("../test-images/1.jpg", "../test-images/1.png", mask_colors);

    return 0;
}

