#include <iostream>
#include <torch/torch.h>
#include "kvald/model.h"
#include "kvald/trainer.h"
#include "kvald/data_loader.h"
#include "kvald/proof_of_concept.h"
#include <json.hpp>
#include <fstream>
#include <cxxopts.hpp>

int main(int argc, char* argv[]) {
    cxxopts::Options options("KVALD", "Glare Detection and Removal");
    options.add_options()
        ("v,video_path", "Path to the input video file", cxxopts::value<std::string>())
        ("o,output_id", "A unique ID for the output files. Defaults to the video's filename.", cxxopts::value<std::string>()->default_value(""))
        ("c,config", "Path to the config file.", cxxopts::value<std::string>()->default_value("prototypes/config.json"))
        ("train", "Run the training process")
        ("h,help", "Print usage");

    auto result = options.parse(argc, argv);

    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        return 0;
    }

    // Load config.json
    nlohmann::json config;
    std::ifstream config_file(result["config"].as<std::string>());
    if (config_file.is_open()) {
        config_file >> config;
        config_file.close();
    } else {
        std::cerr << "Error: Could not open config.json at " << result["config"].as<std::string>() << std::endl;
        return 1;
    }

    torch::Device device(config["train"]["device"].get<std::string>());

    if (result.count("train")) {
        std::cout << "Running training process..." << std::endl;
        int n_videos = config["train"]["n_videos"];
        int epochs = config["train"]["num_epochs"];
        int batch_size = config["train"]["batch_size"];
        float lr = config["train"]["learning_rate"];
        float loss_weight = config["unsupervised"]["loss_weight"];

        auto dataset = std::make_shared<kvald::GlareDataset>(n_videos);
        kvald::train_model(dataset, epochs, batch_size, lr, device, loss_weight);
    } else if (result.count("video_path")) {
        std::cout << "Running video processing..." << std::endl;
        std::string video_path = result["video_path"].as<std::string>();
        std::string output_id = result["output_id"].as<std::string>();
        if (output_id.empty()) {
            // Extract filename from video_path
            size_t last_slash = video_path.find_last_of("/");
            if (last_slash == std::string::npos) {
                last_slash = video_path.find_last_of("\\");
            }
            std::string filename = (last_slash == std::string::npos) ? video_path : video_path.substr(last_slash + 1);
            size_t dot_pos = filename.find_last_of(".");
            output_id = (dot_pos == std::string::npos) ? filename : filename.substr(0, dot_pos);
        }

        kvald::UNet model;
        try {
            torch::load(model, "kvald_unet.pt");
        } catch (const c10::Error& e) {
            std::cerr << "Error loading model: " << e.what() << std::endl;
            std::cerr << "Please ensure 'kvald_unet.pt' exists in the current directory after training." << std::endl;
            return 1;
        }

        try {
            kvald::video_processing(video_path, output_id, model, config);
        } catch (const kvald::VideoProcessingError& e) {
            std::cerr << "Video processing error: " << e.what() << std::endl;
            return 1;
        } catch (const std::exception& e) {
            std::cerr << "An unexpected error occurred during video processing: " << e.what() << std::endl;
            return 1;
        }
    } else {
        std::cout << "No action specified. Use --train to train or --video_path to process a video." << std::endl;
        std::cout << options.help() << std::endl;
    }

    return 0;
}