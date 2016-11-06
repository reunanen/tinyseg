#include "../tinyseg.hpp"

void test()
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
#if SIZE_MAX > 0xffffffff
        // 64-bit build
        tinyseg::load_image("../test-images/02.jpg", "../test-images/02_labels.png", label_colors),
#endif
    };

    std::cout << samples.size() << " images read" << std::endl;

    tinyseg::create_training_dataset_params create_training_dataset_params;

    tiny_dnn::network<tiny_dnn::sequential> net;

    const int input_width = 2 * create_training_dataset_params.window_size_half.width + 1;
    const int input_height = 2 * create_training_dataset_params.window_size_half.height + 1;
    const tiny_dnn::cnn_size_t class_count = static_cast<tiny_dnn::cnn_size_t>(label_colors.size());
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

    const size_t minibatch_size = 100;
    const size_t max_epoch_count = 100;
    const size_t early_stop_count = 10;

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
        {
            const auto training_set = tinyseg::create_training_dataset(samples.begin(), samples.end(), create_training_dataset_params);

            const auto on_epoch_enumerate = []() {};
            const auto on_batch_enumerate = []() {};

            net.train<tiny_dnn::mse>(optimizer, training_set.inputs, training_set.labels, minibatch_size, 1, on_batch_enumerate, on_epoch_enumerate, reset_weights);

            optimizer.alpha = std::max(0.5f * optimizer.alpha, min_alpha);
        }

        const auto test_set = tinyseg::create_training_dataset(samples.begin(), samples.end(), create_training_dataset_params);

        const tiny_dnn::result test_result = net.test(test_set.inputs, test_set.labels);

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
        input_filename << "../test-images/" << std::setw(2) << std::setfill('0') << test_image << ".jpg";
        output_filename << "../test-images/" << std::setw(2) << std::setfill('0') << test_image << "_result.png";

        cv::Mat roi; // no ROI

        cv::Mat input_image = cv::imread(input_filename.str(), cv::IMREAD_GRAYSCALE);
        while (input_image.size().area() > 512 * 512) {
            const double resize_factor = 1.0 / sqrt(2.0);
            cv::resize(input_image, input_image, cv::Size(), resize_factor, resize_factor);

            std::ostringstream resized_input_filename;
            resized_input_filename << "../test-images/" << std::setw(2) << std::setfill('0') << test_image << "_resized.jpg";
            cv::imwrite(resized_input_filename.str(), input_image);
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

        std::cout << " " << test_image;
    }

    std::cout << " - Done!" << std::endl;
}

int main(int argc, char* argv[])
{
    try {
        test();
    }
    catch (const std::exception& e) {
        std::cerr << std::endl << e.what() << std::endl;
    }
    
    return 0;
}