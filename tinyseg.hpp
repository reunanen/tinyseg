#include <tiny_dnn/tiny_dnn.h>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

namespace tinyseg {

struct dataset {
    std::vector<tiny_dnn::vec_t> inputs;
    std::vector<tiny_dnn::label_t> labels;
    std::vector<tiny_dnn::vec_t> weights;
};

dataset load_image(const std::string& input_file_name, const std::string& mask_file_name, const std::vector<cv::Scalar>& mask_colors) {
    const cv::Mat_<uint8_t> input = cv::imread(input_file_name, cv::IMREAD_GRAYSCALE);
    const cv::Mat mask = cv::imread(mask_file_name);
    cv::Mat_<uint16_t> labels(input.size(), 0);
    cv::Mat_<uint8_t> weights(input.size(), 0);

    cv::Mat label_mask;
    for (tiny_dnn::label_t label = 0, label_count = mask_colors.size(); label < label_count; ++label) {
        const cv::Scalar& mask_color = mask_colors[label];
        cv::inRange(mask, mask_color, mask_color, label_mask);
        labels.setTo(label, label_mask);
        weights.setTo(1, label_mask);
    }

    dataset dataset;

    std::transform(input.begin(), input.end(), std::back_inserter(dataset.inputs), [](uint8_t input) { return static_cast<tiny_dnn::float_t>(input); });
    std::transform(labels.begin(), labels.end(), std::back_inserter(dataset.labels), [](uint16_t label) { return static_cast<tiny_dnn::float_t>(label); });
    std::transform(weights.begin(), weights.end(), std::back_inserter(dataset.weights), [](uint8_t weight) { return static_cast<tiny_dnn::float_t>(weight); });

    return dataset;
}

}