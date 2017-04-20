#include "tinyseg.hpp"
#include "tinyseg-net-structure.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <numeric> // std::iota
#include <random>

namespace tinyseg {

void training_dataset::shuffle() {
    if (!is_valid()) {
        throw std::runtime_error("Training dataset is not valid");
    }
    std::vector<size_t> new_order(inputs.size());
    std::iota(new_order.begin(), new_order.end(), 0);
    std::shuffle(new_order.begin(), new_order.end(), std::mt19937{ std::random_device{}() });

    std::vector<image_t> new_inputs(inputs.size());
    std::vector<label_matrix_t> new_labels(labels.size());

    for (size_t i = 0, end = inputs.size(); i < end; ++i) {
        new_inputs[i] = inputs[new_order[i]];
        new_labels[i] = labels[new_order[i]];
    }

    std::swap(inputs, new_inputs);
    std::swap(labels, new_labels);
}

std::pair<training_dataset, training_dataset> training_dataset::split(double first_percentage) {
    std::pair<training_dataset, training_dataset> result;
    training_dataset shuffled = *this;
    shuffled.shuffle();
    size_t first_count = static_cast<size_t>(inputs.size() * first_percentage / 100.0);
    std::move(shuffled.inputs.begin(), shuffled.inputs.begin() + first_count, std::back_inserter(result.first.inputs));
    std::move(shuffled.labels.begin(), shuffled.labels.begin() + first_count, std::back_inserter(result.first.labels));
    std::move(shuffled.inputs.begin() + first_count, shuffled.inputs.end(), std::back_inserter(result.second.inputs));
    std::move(shuffled.labels.begin() + first_count, shuffled.labels.end(), std::back_inserter(result.second.labels));
    return result;
}

bool training_dataset::is_valid() {
    return inputs.size() == labels.size();
}

sample load_image(const std::string& original_image_filename, const std::string& labels_filename, const std::vector<cv::Scalar>& label_colors, int original_image_read_flags, int label_image_read_flags) {
    sample sample;

    sample.original_image = cv::imread(original_image_filename, original_image_read_flags);

    if (sample.original_image.data == nullptr) {
        throw std::runtime_error("Unable to read original image file");
    }

    const cv::Mat labels_mask = cv::imread(labels_filename, label_image_read_flags);

    if (labels_mask.data == nullptr) {
        throw std::runtime_error("Unable to read labels image file");
    }

    cv::Size image_size = sample.original_image.size();

    if (labels_mask.size() != image_size) {
        throw std::runtime_error("Original and labels image size mismatch");
    }

    sample.labels.create(image_size, label_image_type);
    sample.labels.setTo(dlib::label_to_ignore);

    cv::Mat label_mask;
    for (label_image_t label = 0, label_count = static_cast<label_image_t>(label_colors.size()); label < label_count; ++label) {
        const cv::Scalar& label_color = label_colors[label];
        cv::inRange(labels_mask, label_color, label_color, label_mask);
        sample.labels.setTo(label, label_mask);
    }

    return sample;
}

cv::Mat make_border(const cv::Mat& input, const create_training_dataset_params& params) {
    if (input.data == nullptr) {
        throw std::runtime_error("make_border: image is empty");
    }
    cv::Mat output;
    cv::copyMakeBorder(input, output,
        border_required, border_required,
        border_required, border_required,
        params.border_type, params.border_value);
    return output;
}

image_t convert_to_dlib_input(const cv::Mat& original_image, const cv::Mat& roi, const create_training_dataset_params& params) {

    cv::Mat scaled_image(200, 200, original_image.type());

    cv::resize(original_image, scaled_image, scaled_image.size(), 0.0, 0.0, cv::INTER_LINEAR);

    image_t result(scaled_image.rows, scaled_image.cols);

    to_dlib_matrix(cv::Mat_<uint8_t>(scaled_image), result);

    return result;
}

}
