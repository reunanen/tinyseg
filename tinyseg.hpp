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

struct sample {
    cv::Mat original_image;
    cv::Mat labels;
    cv::Mat mask; // non-zero where labels are valid
};

sample load_image(const std::string& original_image_filename, const std::string& labels_filename, const std::vector<cv::Scalar>& label_colors) {
    sample sample;

    sample.original_image = cv::imread(original_image_filename, cv::IMREAD_GRAYSCALE);

    if (sample.original_image.data == nullptr) {
        throw std::runtime_error("Unable to read original image file");
    }

    const cv::Mat labels_mask = cv::imread(labels_filename);

    if (labels_mask.data == nullptr) {
        throw std::runtime_error("Unable to read labels image file");
    }

    cv::Size image_size = sample.original_image.size();

    if (labels_mask.size() != image_size) {
        throw std::runtime_error("Original and labels image size mismatch");
    }

    sample.labels.create(image_size, CV_16UC1);
    sample.labels.setTo(std::numeric_limits<unsigned short>::max());

    sample.mask.create(image_size, CV_8UC1);
    sample.mask.setTo(0);

    cv::Mat label_mask;
    for (tiny_dnn::label_t label = 0, label_count = label_colors.size(); label < label_count; ++label) {
        const cv::Scalar& label_color = label_colors[label];
        cv::inRange(labels_mask, label_color, label_color, label_mask);
        sample.labels.setTo(label, label_mask);
        sample.mask.setTo(255, label_mask);
    }

    return sample;
}

template <typename InputIterator>
dataset create_dataset(InputIterator begin, InputIterator end) {
    dataset dataset;

    std::transform(input.begin(), input.end(), std::back_inserter(dataset.inputs), [](uint8_t input) { return static_cast<tiny_dnn::float_t>(input); });
    std::transform(labels.begin(), labels.end(), std::back_inserter(dataset.labels), [](uint16_t label) { return static_cast<tiny_dnn::float_t>(label); });
    std::transform(weights.begin(), weights.end(), std::back_inserter(dataset.weights), [](uint8_t weight) { return static_cast<tiny_dnn::float_t>(weight); });

    return dataset;

}

}