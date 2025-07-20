#include "kvald/synthetic_data.h"
#include <cmath>
#include <numeric>
#include <algorithm>

namespace kvald
{

Eigen::MatrixXf create_glare(
    int H,
    int W,
    const std::pair<int, int>& center,
    float sigma
)
{
    Eigen::MatrixXf g(H, W);
    int cy = center.first;
    int cx = center.second;

    for (int y = 0; y < H; ++y)
    {
        for (int x = 0; x < W; ++x)
        {
            g(y, x) = std::exp(-((std::pow(y - cy, 2) + std::pow(x - cx, 2)) / (2 * std::pow(sigma, 2))));
        }
    }
    return g / g.maxCoeff();
}

std::pair<std::vector<Eigen::MatrixXf>, std::vector<Eigen::MatrixXf>> static_glare(
    const std::vector<Eigen::MatrixXf>& background,
    std::mt19937& rng,
    float intensity,
    int sigma
)
{
    int T = background.size();
    if (T == 0) return {{}, {}};
    int H = background[0].rows();
    int W = background[0].cols();

    std::uniform_int_distribution<> dist_y(sigma, H - sigma - 1);
    std::uniform_int_distribution<> dist_x(sigma, W - sigma - 1);

    int cy = dist_y(rng);
    int cx = dist_x(rng);

    Eigen::MatrixXf spot = create_glare(H, W, {cy, cx}, sigma);

    std::vector<Eigen::MatrixXf> mask(T);
    std::vector<Eigen::MatrixXf> video(T);

    for (int t = 0; t < T; ++t)
    {
        mask[t] = spot.cast<float>();
        video[t] = (background[t].array() + intensity * mask[t].array()).cwiseMin(1.0f).cwiseMax(0.0f).matrix();
    }
    return {video, mask};
}

std::pair<std::vector<Eigen::MatrixXf>, std::vector<Eigen::MatrixXf>> pulsing_glare(
    const std::vector<Eigen::MatrixXf>& background,
    std::mt19937& rng,
    float intensity,
    int sigma
)
{
    int T = background.size();
    if (T == 0) return {{}, {}};
    int H = background[0].rows();
    int W = background[0].cols();

    std::uniform_int_distribution<> dist_y(sigma, H - sigma - 1);
    std::uniform_int_distribution<> dist_x(sigma, W - sigma - 1);
    std::uniform_int_distribution<> dist_period(20, 60);

    int cy = dist_y(rng);
    int cx = dist_x(rng);
    Eigen::MatrixXf spot = create_glare(H, W, {cy, cx}, sigma);
    int period = dist_period(rng);

    std::vector<Eigen::MatrixXf> mask(T);
    std::vector<Eigen::MatrixXf> video(T);

    for (int t = 0; t < T; ++t)
    {
        float alpha = 0.5f * (1.0f + std::sin(2 * M_PI * t / period));
        mask[t] = spot * alpha;
        video[t] = (background[t].array() + intensity * mask[t].array()).cwiseMin(1.0f).cwiseMax(0.0f).matrix();
    }
    return {video, mask};
}

std::pair<std::vector<Eigen::MatrixXf>, std::vector<Eigen::MatrixXf>> traveling_glare(
    const std::vector<Eigen::MatrixXf>& background,
    std::mt19937& rng,
    float intensity,
    int sigma
)
{
    int T = background.size();
    if (T == 0) return {{}, {}};
    int H = background[0].rows();
    int W = background[0].cols();

    std::uniform_int_distribution<> dist_y(sigma, H - sigma - 1);
    std::uniform_int_distribution<> dist_x(sigma, W - sigma - 1);

    std::pair<int, int> start = {dist_y(rng), dist_x(rng)};
    std::pair<int, int> end = {dist_y(rng), dist_x(rng)};

    std::vector<Eigen::MatrixXf> mask(T);
    std::vector<Eigen::MatrixXf> video(T);

    for (int t = 0; t < T; ++t)
    {
        float frac = static_cast<float>(t) / (T - 1.0f);
        int cy = static_cast<int>(start.first + frac * (end.first - start.first));
        int cx = static_cast<int>(start.second + frac * (end.second - start.second));
        Eigen::MatrixXf spot = create_glare(H, W, {cy, cx}, sigma);
        mask[t] = spot;
        video[t] = (background[t].array() + intensity * spot.array()).cwiseMin(1.0f).cwiseMax(0.0f).matrix();
    }
    return {video, mask};
}

std::pair<std::map<std::string, std::vector<Eigen::MatrixXf>>, std::map<std::string, std::vector<Eigen::MatrixXf>>> generate_sample_videos(
    int seed,
    int T,
    int H,
    int W,
    float mean,
    float std
)
{
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal_dist(mean, std);

    std::vector<Eigen::MatrixXf> background(T);
    for (int t = 0; t < T; ++t)
    {
        background[t].resize(H, W);
        for (int r = 0; r < H; ++r)
        {
            for (int c = 0; c < W; ++c)
            {
                background[t](r, c) = std::min(1.0f, std::max(0.0f, normal_dist(rng)));
            }
        }
    }

    std::map<std::string, std::function<std::pair<std::vector<Eigen::MatrixXf>, std::vector<Eigen::MatrixXf>>(
        const std::vector<Eigen::MatrixXf>&, std::mt19937&, float, int)>> glare_funcs;
    glare_funcs["static"] = static_glare;
    glare_funcs["pulsing"] = pulsing_glare;
    glare_funcs["traveling"] = traveling_glare;

    std::string style = "pulsing"; // For consistent testing
    std::uniform_real_distribution<float> uniform_dist(0.5f, 1.0f);
    float intensity = uniform_dist(rng);
    std::uniform_int_distribution<> sigma_dist(1, 5);
    int sigma = sigma_dist(rng);

    auto result = glare_funcs[style](background, rng, intensity, sigma);
    
    std::map<std::string, std::vector<Eigen::MatrixXf>> video_dict;
    std::map<std::string, std::vector<Eigen::MatrixXf>> mask_dict;

    video_dict[style] = result.first;
    mask_dict[style] = result.second;

    return {video_dict, mask_dict};
}

} // namespace kvald