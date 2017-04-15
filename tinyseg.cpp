#include "tinyseg.hpp"
#include "tinyseg-net-structure.hpp"

#include <opencv2/imgcodecs.hpp>

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
    std::vector<label_t> new_labels(labels.size());

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

    sample.labels.create(image_size, label_image_type);
    sample.labels.setTo(std::numeric_limits<label_image_t>::max());

    std::vector<cv::Point> mask;

    cv::Mat label_mask;
    for (label_image_t label = 0, label_count = static_cast<label_image_t>(label_colors.size()); label < label_count; ++label) {
        const cv::Scalar& label_color = label_colors[label];
        cv::inRange(labels_mask, label_color, label_color, label_mask);
        sample.labels.setTo(label, label_mask);

        mask.clear();
        cv::findNonZero(label_mask, mask);
        std::move(mask.begin(), mask.end(), std::back_inserter(sample.mask));
    }

    return sample;
}

cv::Mat make_border(const cv::Mat& original_image, const create_training_dataset_params& params) {
    if (original_image.data == nullptr) {
        throw std::runtime_error("make_border: original_image is empty");
    }
    cv::Mat original_image_with_borders;
    const size_t input_image_size_half = input_image_size / 2;
    cv::copyMakeBorder(original_image, original_image_with_borders,
        input_image_size_half, input_image_size_half,
        input_image_size_half, input_image_size_half,
        params.border_type, params.border_value);
    return original_image_with_borders;
}

void to_dlib_matrix(const cv::Mat_<uint8_t>& input, image_t& output)
{
    assert(input.rows == output.nr());
    assert(input.cols == output.nc());
    assert(input.type() == CV_8UC1);

    for (int y = 0; y < input.rows; ++y) {
        const uint8_t* input_row = input.ptr(y);
        for (int x = 0; x < input.cols; ++x) {
            const uint8_t value = input_row[x];
            //output(y, x) = dlib::rgb_pixel(value, value, value);
            output(y, x) = value;
        }
    }
}

#if 0
template <typename InputIterator>
training_dataset create_training_dataset(InputIterator begin, InputIterator end, const create_training_dataset_params& params = create_training_dataset_params()) {
    training_dataset dataset;

    const size_t initial_capacity = 16 * 1024 * 1024;
    dataset.inputs.reserve(initial_capacity);
    dataset.labels.reserve(initial_capacity);

#ifdef _DEBUG
    size_t sample_index = 0;
#endif // _DEBUG

    for (InputIterator i = begin; i != end; ++i) {
        const sample& sample = *i;

        const cv::Mat original_image_with_borders = make_border(sample.original_image, params);

        image_t input(input_image_size, input_image_size); // , weights;

        for (const auto& point : sample.mask) {

#ifdef _DEBUG
            if (sample_index++ % 100 != 0) {
                continue;
            }
#endif // _DEBUG

            cv::Rect source_rect(point.x, point.y, input_image_size, input_image_size);
            cv::Mat_<uint8_t> input_window = original_image_with_borders(source_rect);

            to_dlib_matrix(input_window, input);

            label_t label = sample.labels.at<label_image_t>(point);

            dataset.inputs.push_back(input);
            dataset.labels.push_back(label);
        }
    }

    return dataset;
}
#endif

std::vector<image_t> convert_to_dlib_inputs(const cv::Mat& original_image, const cv::Mat& roi, const create_training_dataset_params& params) {
    std::vector<cv::Point> nz;
    if (roi.data == nullptr) { // no ROI given
        nz.reserve(original_image.size().area());
        for (int y = 0; y < original_image.rows; ++y) {
            for (int x = 0; x < original_image.cols; ++x) {
                nz.push_back(cv::Point(x, y));
            }
        }
    }
    else {
        cv::findNonZero(roi, nz);
    }

    const cv::Mat original_image_with_borders = make_border(original_image, params);

    std::vector<image_t> result;
    result.reserve(nz.size());

    image_t input(input_image_size, input_image_size);

    for (const auto& point : nz) {
        cv::Rect source_rect(point.x, point.y, input_image_size, input_image_size);
        const cv::Mat_<uint8_t> input_window = original_image_with_borders(source_rect);

        to_dlib_matrix(input_window, input);
        result.push_back(input);
    }

    return result;
}

}
