#include <tiny_dnn/tiny_dnn.h>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

namespace tinyseg {

struct dataset {
    std::vector<tiny_dnn::label_t> images;
    std::vector<tiny_dnn::vec_t> labels;
};

dataset load_image(const std::string& input_file_name, const std::string& mask_file_name, std::vector<cv::Scalar> mask_colors) {
    const cv::Mat input = cv::imread(input_file_name);
    const cv::Mat mask = cv::imread(mask_file_name);

    cv::Mat label_mask;
    for (tiny_dnn::label_t label = 0, label_count = mask_colors.size(); label < label_count; ++label) {
        const cv::Scalar& mask_color = mask_colors[label];
        cv::inRange(mask, mask_color, mask_color, label_mask);
    }

    dataset dataset;
    return dataset;
}

}