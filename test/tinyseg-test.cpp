#pragma warning( disable : 4503 ) // disable compiler warning C4503: decorated name length exceeded, name was truncated

#include "../tinyseg.hpp"
#include "../tinyseg-net-structure.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

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

    std::cout << "Creating the training dataset: ";

    tinyseg::create_training_dataset_params create_training_dataset_params;

    auto& full_dataset = tinyseg::create_training_dataset(samples.begin(), samples.end(), create_training_dataset_params);

    std::cout << full_dataset.inputs.size() << " samples created" << std::endl;

    tinyseg::net_type net;

    const size_t minibatch_size = 2000;
    const double initial_learning_rate = 1e-5;
    const double weight_decay = 0.0005;
    const double momentum = 0.9;

    dlib::dnn_trainer<tinyseg::net_type> trainer(net, dlib::sgd(weight_decay, momentum));

    trainer.be_verbose();
    trainer.set_learning_rate(initial_learning_rate);
    //trainer.set_synchronization_file("tinyseg-test-state.dat", std::chrono::minutes(10));
    trainer.set_iterations_without_progress_threshold(10);
    trainer.set_learning_rate_shrink_factor(0.1);

    tinyseg::training_dataset minibatch;

    std::cout << "Training:" << std::endl;

    while (trainer.get_learning_rate() >= 1e-5) {

#if 0
        minibatch.inputs.clear();
        minibatch.labels.clear();

        for (size_t i = 0; i < minibatch_size; ++i) {
            size_t index = rand() % full_dataset.inputs.size();
            minibatch.inputs.push_back(full_dataset.inputs[index]);
            minibatch.labels.push_back(full_dataset.labels[index]);
        }

        trainer.train_one_step(minibatch.inputs, minibatch.labels);
#endif
        trainer.train(full_dataset.inputs, full_dataset.labels);

    }

    trainer.get_net();
    net.clean();

    tinyseg::runtime_net_type runtime_net = net;

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

        auto test_input = convert_to_dlib_input(input_image, roi, create_training_dataset_params);

        assert(test_input.nr() == input_image.rows);
        assert(test_input.nc() == input_image.cols);

        const dlib::matrix<tinyseg::label_t> predicted_labels = runtime_net(test_input);

        assert(test_input.nr() == predicted_labels.nr());
        assert(test_input.nc() == predicted_labels.nc());

        for (int y = 0; y < input_image.rows; ++y) {
            for (int x = 0; x < input_image.cols; ++x) {
                const tinyseg::label_t label = predicted_labels(y, x);
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
