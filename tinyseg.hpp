#include <tiny_dnn/tiny_dnn.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <numeric> // std::iota

namespace tinyseg {

typedef unsigned short label_image_t;
const int label_image_type = CV_16UC1;

struct training_dataset {
    std::vector<tiny_dnn::vec_t> inputs;
    std::vector<tiny_dnn::label_t> labels;
    //std::vector<tiny_dnn::vec_t> weights;

    void shuffle() {
        if (!is_valid()) {
            throw std::runtime_error("Training dataset is not valid");
        }
        std::vector<size_t> new_order(inputs.size());
        std::iota(new_order.begin(), new_order.end(), 0);
        std::shuffle(new_order.begin(), new_order.end(), std::mt19937{ std::random_device{}() });

        std::vector<tiny_dnn::vec_t> new_inputs(inputs.size());
        std::vector<tiny_dnn::label_t> new_labels(labels.size());

        for (size_t i = 0, end = inputs.size(); i < end; ++i) {
            new_inputs[i] = inputs[new_order[i]];
            new_labels[i] = labels[new_order[i]];
        }

        std::swap(inputs, new_inputs);
        std::swap(labels, new_labels);
    }

    std::pair<training_dataset, training_dataset> split(double first_percentage = 50.0) {
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

    bool is_valid() {
        return inputs.size() == labels.size();
    }
};

struct sample {
    cv::Mat original_image;
    cv::Mat labels;
    std::vector<cv::Point> mask; // where are the labels valid?
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

    std::vector<cv::Point> mask;

    cv::Mat label_mask;
    for (tiny_dnn::label_t label = 0, label_count = static_cast<tiny_dnn::label_t>(label_colors.size()); label < label_count; ++label) {
        const cv::Scalar& label_color = label_colors[label];
        cv::inRange(labels_mask, label_color, label_color, label_mask);
        sample.labels.setTo(label, label_mask);

        mask.clear();
        cv::findNonZero(label_mask, mask);
        std::move(mask.begin(), mask.end(), std::back_inserter(sample.mask));
    }

    return sample;
}

struct create_training_dataset_params {
    cv::Size window_size_half = cv::Size(20, 20);
    int border_type = cv::BORDER_REFLECT;
    cv::Scalar border_value = cv::Scalar();

    uint32_t sample_count = 32768;
};

cv::Mat make_border(const cv::Mat& original_image, const create_training_dataset_params& params) {
    if (original_image.data == nullptr) {
        throw std::runtime_error("make_border: original_image is empty");
    }
    cv::Mat original_image_with_borders;
    cv::copyMakeBorder(original_image, original_image_with_borders,
        params.window_size_half.height, params.window_size_half.height,
        params.window_size_half.width, params.window_size_half.width,
        params.border_type, params.border_value);
    return original_image_with_borders;
}

template <typename InputIterator>
training_dataset create_training_dataset(InputIterator begin, InputIterator end, const create_training_dataset_params& params = create_training_dataset_params()) {
    training_dataset dataset;

    typedef unsigned long long index_type;

    const index_type total_available_sample_count = std::accumulate(begin, end, static_cast<index_type>(0), [](index_type result, const sample& sample) { return result + sample.mask.size(); });

    std::vector<index_type> sample_indexes(params.sample_count);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<index_type> distribution(0, total_available_sample_count);

    for (size_t picked_sample = 0; picked_sample < params.sample_count; ++picked_sample) {
        sample_indexes[picked_sample] = distribution(gen); // with replacement
    }

    std::sort(sample_indexes.begin(), sample_indexes.end());

    dataset.inputs.reserve(params.sample_count);
    dataset.labels.reserve(params.sample_count);

    index_type sample_index = 0;
    uint32_t picked_sample_index = 0;

    for (InputIterator i = begin; i != end && picked_sample_index < params.sample_count; ++i) {
        const sample& sample = *i;

        // TODO: avoid re-doing this over and over again
        const cv::Mat original_image_with_borders = make_border(sample.original_image, params);

        tiny_dnn::vec_t input; // , weights;

        for (size_t point_index = 0, end = sample.mask.size(); point_index < end && picked_sample_index < params.sample_count; ) {
            const auto& point = sample.mask[point_index];
            if (sample_index == sample_indexes[picked_sample_index]) {
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

                ++picked_sample_index;
            }
            else {
                ++sample_index;
                ++point_index;
            }
        }
    }

    assert(dataset.inputs.size() == params.sample_count);
    assert(dataset.labels.size() == params.sample_count);

    dataset.shuffle();

    return dataset;
}

struct runtime_params {
    // TODO: sampling density, etc
};

tiny_dnn::tensor_t convert_to_tinydnn_inputs(const cv::Mat& original_image, const cv::Mat& roi = cv::Mat(), const create_training_dataset_params& params = create_training_dataset_params()) {
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

    cv::Mat float_image;
    original_image_with_borders.convertTo(float_image, CV_32FC1);

    tiny_dnn::tensor_t result;
    result.reserve(nz.size());

    for (const auto& point : nz) {
        cv::Rect source_rect(point.x, point.y, params.window_size_half.width * 2 + 1, params.window_size_half.height * 2 + 1);
        cv::Mat_<float> input_window = float_image(source_rect);
        tiny_dnn::vec_t input(input_window.begin(), input_window.end());
        result.push_back(input);
    }

    return result;
}

}