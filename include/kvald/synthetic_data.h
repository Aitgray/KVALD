#pragma once

#include <Eigen/Dense>
#include <vector>
#include <random>
#include <map>

namespace kvald
{

Eigen::MatrixXf create_glare(
    int H,
    int W,
    const std::pair<int, int>& center,
    float sigma
);

std::pair<std::vector<Eigen::MatrixXf>, std::vector<Eigen::MatrixXf>> static_glare(
    const std::vector<Eigen::MatrixXf>& background,
    std::mt19937& rng,
    float intensity,
    int sigma
);

std::pair<std::vector<Eigen::MatrixXf>, std::vector<Eigen::MatrixXf>> pulsing_glare(
    const std::vector<Eigen::MatrixXf>& background,
    std::mt19937& rng,
    float intensity,
    int sigma
);

std::pair<std::vector<Eigen::MatrixXf>, std::vector<Eigen::MatrixXf>> traveling_glare(
    const std::vector<Eigen::MatrixXf>& background,
    std::mt19937& rng,
    float intensity,
    int sigma
);

std::pair<std::map<std::string, std::vector<Eigen::MatrixXf>>, std::map<std::string, std::vector<Eigen::MatrixXf>>> generate_sample_videos(
    int seed,
    int T = 150,
    int H = 128,
    int W = 128,
    float mean = 0.5f,
    float std = 0.1f
);

} // namespace kvald