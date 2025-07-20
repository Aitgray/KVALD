#pragma once

#include <torch/torch.h>
#include <vector>
#include "kvald/synthetic_data.h"

namespace kvald
{

class GlareDataset : public torch::data::Dataset<GlareDataset>
{
public:
    GlareDataset(int n_videos);

    torch::data::Example<> get(size_t index) override;

    torch::optional<size_t> size() const override;

private:
    std::vector<std::pair<torch::Tensor, torch::Tensor>> samples;
};

} // namespace kvald