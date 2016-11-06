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
        tinyseg::load_image("../test-images/01.jpg", "../test-images/01_labels.png", label_colors),
        //tinyseg::load_image("../test-images/02.jpg", "../test-images/02_labels.png", label_colors),
    };

    std::cout << samples.size() << " images read" << std::endl;

    std::cout << "Creating the training dataset: ";

    tinyseg::create_training_dataset_params create_training_dataset_params;

    auto& full_dataset = tinyseg::create_training_dataset(samples.begin(), samples.end(), create_training_dataset_params);

    std::cout << full_dataset.inputs.size() << " samples created" << std::endl;

    tiny_dnn::network<tiny_dnn::sequential> net;

    const int input_width = 2 * create_training_dataset_params.window_size_half.width + 1;
    const int input_height = 2 * create_training_dataset_params.window_size_half.height + 1;
    const int class_count = label_colors.size();
    const int initial_conv = 6;
    const int feature_map_count = 6;
    const int pooling = 2;
    const int fully_connected_neuron_count = 40;

    // add layers
    net << tiny_dnn::conv<tiny_dnn::tan_h>(input_width, input_height, initial_conv, 1, feature_map_count)
        << tiny_dnn::ave_pool<tiny_dnn::tan_h>(input_width - initial_conv + 1, input_width - initial_conv + 1, feature_map_count, pooling)
        << tiny_dnn::fc<tiny_dnn::tan_h>((input_width - initial_conv + 1) / pooling * (input_width - initial_conv + 1) / pooling * feature_map_count, fully_connected_neuron_count)
        << tiny_dnn::fc<tiny_dnn::identity>(fully_connected_neuron_count, class_count);

    assert(net.in_data_size() == input_width * input_height);
    assert(net.out_data_size() == class_count);

    // declare optimization algorithm
    tiny_dnn::adagrad optimizer;
    
    const tiny_dnn::float_t min_alpha(0.0000001f);

    const size_t minibatch_size = 50;
    const size_t max_epoch_count = 20;
    const size_t early_stop_count = 3;

    std::cout << "Training for max " << max_epoch_count << " epochs:";

    std::vector<double> test_accuracies;

    const auto early_stop_criterion = [&test_accuracies, &early_stop_count]() {
        const size_t count = test_accuracies.size();
        return count > early_stop_count && test_accuracies[count - 1] <= test_accuracies[count - 1 - early_stop_count];
    };

    auto best_net = net;
    tiny_dnn::float_t best_accuracy = 0.0;

    bool reset_weights = true;

    size_t epoch = 0;

    while (true) {
        std::pair<tinyseg::training_dataset, tinyseg::training_dataset> split_data = full_dataset.split(50.0);
        tinyseg::training_dataset& training_data = split_data.first;
        tinyseg::training_dataset& test_data = split_data.second;

        const auto on_epoch_enumerate = []() {};
        const auto on_batch_enumerate = []() {};

        net.train<tiny_dnn::mse>(optimizer, training_data.inputs, training_data.labels, minibatch_size, 1, on_batch_enumerate, on_epoch_enumerate, reset_weights);

        //std::cout << ++epoch << " ";
        optimizer.alpha = std::max(0.5f * optimizer.alpha, min_alpha);

        const tiny_dnn::result test_result = net.test(test_data.inputs, test_data.labels);

        const auto accuracy = test_result.accuracy();

        test_accuracies.push_back(accuracy);

        std::cout << " " << std::fixed << std::setprecision(1) << accuracy << "%";

        if (accuracy > best_accuracy) {
            best_accuracy = accuracy;
            best_net = net;
        }

        reset_weights = false;

        ++epoch;

        if (early_stop_criterion() || epoch >= max_epoch_count) {
            std::cout << std::endl << std::endl << "Confusion matrix:" << std::endl;
            test_result.print_detail(std::cout);
            break;
        }
    }

    std::cout << std::endl << "Testing:";

    for (int test_image = 1; test_image < 10; ++test_image) {

        std::ostringstream input_filename, output_filename;
        input_filename << "../test-images/" << std::setw << std::setfill('0') << test_image << ".jpg";
        output_filename << "../test-images/" << std::setw << std::setfill('0') << test_image << "_result.png";

        cv::Mat roi; // no ROI

        cv::Mat input_image = cv::imread(input_filename.str(), cv::IMREAD_GRAYSCALE);
        while (input_image.size().area() > 512 * 512) {
            cv::resize(input_image, input_image, cv::Size(), 0.5, 0.5);
        }

        cv::Mat result(input_image.size(), CV_8UC3);

        tiny_dnn::tensor_t test_inputs = convert_to_tinydnn_inputs(input_image, roi, create_training_dataset_params);

        assert(test_inputs.size() == input_image.size().area());

        size_t i = 0;
        for (int y = 0; y < input_image.rows; ++y) {
            for (int x = 0; x < input_image.cols; ++x, ++i) {
                tiny_dnn::label_t label = best_net.predict_label(test_inputs[i]);
                const auto& label_color = label_colors[label];
                result.at<cv::Vec3b>(y, x) = cv::Vec3b(
                    static_cast<unsigned char>(label_color[0]),
                    static_cast<unsigned char>(label_color[1]),
                    static_cast<unsigned char>(label_color[2])
                );
            }
        }

        cv::imwrite(output_filename.str(), result);

        std::cout << " " << test_image << std::endl;
    }

    std::cout << " - Done!" << std::endl;

    return 0;
}

