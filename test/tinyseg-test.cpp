#include "../tinyseg.hpp"

int main(int argc, char* argv[])
{
    const std::vector<cv::Scalar> label_colors = {
        cv::Scalar(127, 127, 127),
        cv::Scalar(255, 0, 0),
        cv::Scalar(0, 255, 0),
        cv::Scalar(0, 0, 255),
    };

    std::cout << "Reading the input data: ";

    const auto& samples = std::deque<tinyseg::sample>{
        tinyseg::load_image("../test-images/01.jpg", "../test-images/01.png", label_colors)
    };

    std::cout << samples.size() << " images read" << std::endl;

    std::cout << "Creating the training dataset: ";

    tinyseg::create_training_dataset_params create_training_dataset_params;

    const auto& dataset = tinyseg::create_training_dataset(samples.begin(), samples.end(), create_training_dataset_params);

    std::cout << dataset.inputs.size() << " samples created" << std::endl;

    tiny_dnn::network<tiny_dnn::sequential> net;

    const int input_width = 2 * create_training_dataset_params.window_size_half.width + 1;
    const int input_height = 2 * create_training_dataset_params.window_size_half.height + 1;
    const int class_count = label_colors.size();
    const int initial_conv = 6;
    const int feature_map_count = 6;
    const int pooling = 2;
    const int fully_connected_neuron_count = 120;

    // add layers
    net << tiny_dnn::conv<tiny_dnn::tan_h>(input_width, input_height, initial_conv, 1, feature_map_count)
        << tiny_dnn::ave_pool<tiny_dnn::tan_h>(input_width - initial_conv + 1, input_width - initial_conv + 1, feature_map_count, pooling)
        << tiny_dnn::fc<tiny_dnn::tan_h>((input_width - initial_conv + 1) / pooling * (input_width - initial_conv + 1) / pooling * feature_map_count, fully_connected_neuron_count)
        << tiny_dnn::fc<tiny_dnn::identity>(fully_connected_neuron_count, class_count);

    assert(net.in_data_size() == input_width * input_height);
    assert(net.out_data_size() == class_count);

    // declare optimization algorithm
    tiny_dnn::adagrad optimizer;

    // train (1-epoch, 30-minibatch)
    const size_t minibatch_size = 30;
    const int epoch_count = 2;
    net.train<tiny_dnn::mse>(optimizer, dataset.inputs, dataset.labels, minibatch_size, epoch_count);

    return 0;
}

