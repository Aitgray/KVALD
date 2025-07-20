#include "kvald/data_loader.h"
#include <iostream>

namespace kvald
{

GlareDataset::GlareDataset(int n_videos)
{
    for (int seed = 0; seed < n_videos; ++seed)
    {
        auto sample_videos = generate_sample_videos(seed);
        auto video_dict = sample_videos.first;
        auto mask_dict = sample_videos.second;

        // Assuming 'pulsing' style for now, as in synthetic_data.py
        std::string style = "pulsing";
        const auto& video_frames = video_dict.at(style);
        const auto& mask_frames = mask_dict.at(style);

        for (size_t t = 0; t < video_frames.size(); ++t)
        {
            // Convert Eigen::MatrixXf to torch::Tensor
            // Eigen::MatrixXf is column-major by default, torch::Tensor is row-major.
            // Need to transpose or handle strides carefully.
            // For a 2D matrix, direct copy should work if dimensions are correct.
            torch::Tensor frame_tensor = torch::from_blob(
                video_frames[t].data(), 
                {video_frames[t].rows(), video_frames[t].cols()}, 
                torch::TensorOptions().dtype(torch::kFloat32)
            ).clone(); // .clone() to make it owned data

            torch::Tensor mask_tensor = torch::from_blob(
                mask_frames[t].data(), 
                {mask_frames[t].rows(), mask_frames[t].cols()}, 
                torch::TensorOptions().dtype(torch::kFloat32)
            ).clone();

            // Add a channel dimension (1, H, W)
            frame_tensor = frame_tensor.unsqueeze(0);
            mask_tensor = mask_tensor.unsqueeze(0);

            samples.push_back({frame_tensor, mask_tensor});
        }
    }
}

torch::data::Example<> GlareDataset::get(size_t index)
{
    return {samples[index].first, samples[index].second};
}

torch::optional<size_t> GlareDataset::size() const
{
    return samples.size();
}

} // namespace kvald