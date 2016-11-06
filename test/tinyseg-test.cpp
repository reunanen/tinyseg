#include "../tinyseg.hpp"

int main(int argc, char* argv[])
{
    const std::vector<cv::Scalar> mask_colors = {
        cv::Scalar(127, 127, 127),
        cv::Scalar(255, 0, 0),
        cv::Scalar(0, 255, 0),
        cv::Scalar(0, 0, 255),
    };

    const auto& samples = std::deque<tinyseg::sample>{
        tinyseg::load_image("../test-images/01.jpg", "../test-images/01.png", mask_colors)
    };

    const auto& dataset = tinyseg::create_dataset(samples.begin(), samples.end());

    return 0;
}

