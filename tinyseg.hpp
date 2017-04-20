#ifndef TINYSEG_HPP
#define TINYSEG_HPP

#include <dlib/dnn.h>
#include <opencv2/core.hpp>

namespace tinyseg {

typedef unsigned short label_image_t;
const int label_image_type = CV_16UC1;

typedef uint16_t label_t;

typedef dlib::matrix<dlib::rgb_pixel> image_t;
typedef dlib::matrix<label_t> label_matrix_t;

struct training_dataset {
    std::vector<image_t> inputs;
    std::vector<label_matrix_t> labels;
    //std::vector<tiny_dnn::vec_t> weights;

#if 0
    void shuffle();
    std::pair<training_dataset, training_dataset> split(double first_percentage = 50.0);
    bool is_valid();
#endif
};

struct sample {
    cv::Mat original_image;
    cv::Mat labels;
};

sample load_image(
    const std::string& original_image_filename,
    const std::string& labels_filename,
    const std::vector<cv::Scalar>& label_colors,
    int original_image_read_flags = 1, // cv::IMREAD_COLOR
    int label_image_read_flags = 1 // cv::IMREAD_COLOR
); 

#if 0
struct create_training_dataset_params {
    int border_type = cv::BORDER_REFLECT;
    cv::Scalar border_value = cv::Scalar();
};

cv::Mat make_border(const cv::Mat& input, const create_training_dataset_params& params);
#endif

template <typename OpenCvPixelType, typename DLibPixelType>
void to_dlib_pixel(const OpenCvPixelType& input, DLibPixelType& output) {
    static_assert(sizeof input == sizeof output, "Input and output sizes need to match");
    output = input;
}

template <>
inline void to_dlib_pixel(const cv::Vec3b& input, dlib::rgb_pixel& output) {
    output.red = input[2];
    output.green = input[1];
    output.blue = input[0];
}

template <typename OpenCvPixelType, typename DLibPixelType>
void to_dlib_matrix(const cv::Mat_<OpenCvPixelType>& input, dlib::matrix<DLibPixelType>& output)
{
    assert(input.rows == output.nr());
    assert(input.cols == output.nc());

    for (int y = 0; y < input.rows; ++y) {
        const OpenCvPixelType* input_row = input.ptr<OpenCvPixelType>(y);
        for (int x = 0; x < input.cols; ++x) {
            const OpenCvPixelType& value = input_row[x];
            to_dlib_pixel(value, output(y, x));
        }
    }
}

struct runtime_params {
    // TODO: sampling density, etc
};

}

#endif // TINYSEG_HPP