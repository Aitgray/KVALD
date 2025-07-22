#include "kvald/data_loader.h"
#include <torch/torch.h>
#include <iostream>

namespace kvald {

GlareDataset::GlareDataset(int n_videos) {
    for (int seed = 0; seed < n_videos; ++seed) {
        auto sample_videos = generate_sample_videos(seed);
        auto& video_dict = sample_videos.first;
        auto& mask_dict  = sample_videos.second;

        // Match whatever style you actually want
        const std::string style = "pulsing";
        const auto& video_frames = video_dict.at(style);
        const auto& mask_frames  = mask_dict.at(style);

        for (size_t t = 0; t < video_frames.size(); ++t) {
            const auto& vf = video_frames[t];
            const auto& mf = mask_frames[t];

            const int64_t rows = vf.rows();
            const int64_t cols = vf.cols();

            // Eigen::MatrixXf is column-major by default.
            // Provide explicit strides so PyTorch interprets it correctly,
            // then clone() to own the memory.
            int64_t sizes[2]   = {rows, cols};
            int64_t strides[2] = {1, rows}; // column-major

            auto frame_tensor = torch::from_blob(
                const_cast<float*>(vf.data()),
                torch::IntArrayRef(sizes, 2),
                torch::IntArrayRef(strides, 2),
                torch::TensorOptions().dtype(torch::kFloat32)
            ).clone(); // now row-major contiguous

            auto mask_tensor = torch::from_blob(
                const_cast<float*>(mf.data()),
                torch::IntArrayRef(sizes, 2),
                torch::IntArrayRef(strides, 2),
                torch::TensorOptions().dtype(torch::kFloat32)
            ).clone();

            // Add channel dimension (C,H,W)
            frame_tensor = frame_tensor.unsqueeze(0);
            mask_tensor  = mask_tensor.unsqueeze(0);

            samples.push_back({frame_tensor, mask_tensor});
        }
    }
}

torch::data::Example<> GlareDataset::get(size_t index) {
    return {samples[index].first, samples[index].second};
}

torch::optional<size_t> GlareDataset::size() const {
    return samples.size();
}

} // namespace kvald
