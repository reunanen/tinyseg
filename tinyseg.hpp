#include <dlib/dnn.h>
#include <opencv2/core.hpp>

namespace tinyseg {

typedef unsigned short label_image_t;
const int label_image_type = CV_16UC1;

typedef unsigned long label_t;

typedef dlib::matrix<unsigned char> image_t;

struct training_dataset {
    std::vector<image_t> inputs;
    std::vector<label_t> labels;
    //std::vector<tiny_dnn::vec_t> weights;

    void shuffle();
    std::pair<training_dataset, training_dataset> split(double first_percentage = 50.0);
    bool is_valid();
};

struct sample {
    cv::Mat original_image;
    cv::Mat labels;
    std::vector<cv::Point> mask; // where are the labels valid?
};

sample load_image(const std::string& original_image_filename, const std::string& labels_filename, const std::vector<cv::Scalar>& label_colors);

struct create_training_dataset_params {
    int border_type = cv::BORDER_REFLECT;
    cv::Scalar border_value = cv::Scalar();
};

cv::Mat make_border(const cv::Mat& original_image, const create_training_dataset_params& params);

void to_dlib_matrix(const cv::Mat_<uint8_t>& input, image_t& output);

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
            if (sample_index++ % 1000 != 0) {
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

struct runtime_params {
    // TODO: sampling density, etc
};

std::vector<image_t> convert_to_dlib_inputs(const cv::Mat& original_image, const cv::Mat& roi = cv::Mat(), const create_training_dataset_params& params = create_training_dataset_params());

}
