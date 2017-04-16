#include <dlib/dnn.h>
#include <opencv2/core.hpp>

namespace tinyseg {

typedef unsigned short label_image_t;
const int label_image_type = CV_16UC1;

typedef dlib::matrixoutput_label_t label_t;

typedef dlib::matrix<unsigned char> image_t;
typedef dlib::matrix<label_t> label_matrix_t;

struct training_dataset {
    std::vector<image_t> inputs;
    std::vector<label_matrix_t> labels;
    //std::vector<tiny_dnn::vec_t> weights;

    void shuffle();
    std::pair<training_dataset, training_dataset> split(double first_percentage = 50.0);
    bool is_valid();
};

struct sample {
    cv::Mat original_image;
    cv::Mat labels;
};

sample load_image(const std::string& original_image_filename, const std::string& labels_filename, const std::vector<cv::Scalar>& label_colors);

struct create_training_dataset_params {
    int border_type = cv::BORDER_REFLECT;
    cv::Scalar border_value = cv::Scalar();
};

cv::Mat make_border(const cv::Mat& input, const create_training_dataset_params& params);

template <typename OpenCvPixelType, typename DLibPixelType>
void to_dlib_matrix(const cv::Mat_<OpenCvPixelType>& input, dlib::matrix<DLibPixelType>& output)
{
    assert(input.rows == output.nr());
    assert(input.cols == output.nc());

    for (int y = 0; y < input.rows; ++y) {
        const OpenCvPixelType* input_row = input.ptr<OpenCvPixelType>(y);
        for (int x = 0; x < input.cols; ++x) {
            const OpenCvPixelType value = input_row[x];
            //output(y, x) = dlib::rgb_pixel(value, value, value);
            output(y, x) = value;
        }
    }
}

template <typename InputIterator>
training_dataset create_training_dataset(InputIterator begin, InputIterator end, const create_training_dataset_params& params = create_training_dataset_params()) {
    training_dataset dataset;

    const size_t initial_capacity = end - begin;
    dataset.inputs.reserve(initial_capacity);
    dataset.labels.reserve(initial_capacity);

#ifdef _DEBUG
    size_t sample_index = 0;
#endif // _DEBUG

    cv::Mat scaled_image(200, 200, begin->original_image.type());
    cv::Mat scaled_labels(200, 200, begin->labels.type());

    cv::Mat scaled_image_with_border;

    for (InputIterator i = begin; i != end; ++i) {
        const sample& sample = *i;

        cv::resize(sample.original_image, scaled_image, scaled_image.size(), 0.0, 0.0, cv::INTER_LINEAR);
        cv::resize(sample.labels, scaled_labels, scaled_labels.size(), 0.0, 0.0, cv::INTER_NEAREST);

        scaled_image_with_border = make_border(scaled_image, params);

        image_t input(scaled_image_with_border.rows, scaled_image_with_border.cols); // , weights;
        to_dlib_matrix(cv::Mat_<uint8_t>(scaled_image_with_border), input);

        label_matrix_t labels(scaled_labels.rows, scaled_labels.cols);
        to_dlib_matrix(cv::Mat_<label_image_t>(scaled_labels), labels);

        dataset.inputs.push_back(input);
        dataset.labels.push_back(labels);
    }

    return dataset;
}

struct runtime_params {
    // TODO: sampling density, etc
};

image_t convert_to_dlib_input(const cv::Mat& original_image, const cv::Mat& roi = cv::Mat(), const create_training_dataset_params& params = create_training_dataset_params());

}
