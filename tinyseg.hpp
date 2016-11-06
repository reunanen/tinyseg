#include <tiny_dnn/tiny_dnn.h>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace tinyseg {

typedef unsigned short label_image_t;
const int label_image_type = CV_16UC1;

struct training_dataset {
    std::vector<tiny_dnn::vec_t> inputs;
    std::vector<tiny_dnn::label_t> labels;
    //std::vector<tiny_dnn::vec_t> weights;
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

    sample.labels.create(image_size, label_image_type);
    sample.labels.setTo(std::numeric_limits<label_image_t>::max());

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

struct create_training_dataset_params {
    cv::Size window_size_half = cv::Size(10, 10);
    int border_type = cv::BORDER_REFLECT;
    cv::Scalar border_value = cv::Scalar();
};

template <typename InputIterator>
training_dataset create_training_dataset(InputIterator begin, InputIterator end, const create_training_dataset_params& params = create_training_dataset_params()) {
    training_dataset dataset;

    const size_t initial_capacity = 16 * 1024 * 1024;
    dataset.inputs.reserve(initial_capacity);
    dataset.labels.reserve(initial_capacity);

    for (InputIterator i = begin; i != end; ++i) {
        const sample& sample = *i;

        cv::Mat original_image_with_borders;
        cv::copyMakeBorder(sample.original_image, original_image_with_borders,
            params.window_size_half.height, params.window_size_half.height,
            params.window_size_half.width, params.window_size_half.width,
            params.border_type, params.border_value);

        std::vector<cv::Point> nz;

        cv::findNonZero(sample.mask, nz);        

        tiny_dnn::vec_t input; // , weights;

        for (const auto& point : nz) {
            cv::Rect source_rect(point.x, point.y, params.window_size_half.width * 2 + 1, params.window_size_half.height * 2 + 1);
            cv::Mat_<uint8_t> input_window = original_image_with_borders(source_rect);

            input.clear();
            std::transform(input_window.begin(), input_window.end(), std::back_inserter(input),
                [](uint8_t input) {
                    return static_cast<tiny_dnn::float_t>(input);
                });

            tiny_dnn::label_t label = sample.labels.at<label_image_t>(point);

            dataset.inputs.push_back(input);
            dataset.labels.push_back(label);
        }
    }

    return dataset;
}

}