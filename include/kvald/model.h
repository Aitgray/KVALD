#pragma once

#include <torch/torch.h>

namespace kvald
{

class DoubleConvImpl : public torch::nn::Module
{
public:
    DoubleConvImpl(int64_t in_channels, int64_t out_channels);
    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Sequential double_conv;
};

TORCH_MODULE(DoubleConv);

class UNetImpl : public torch::nn::Module
{
public:
    UNetImpl(int64_t in_channels = 1, int64_t out_channels = 1, const std::vector<int64_t>& features = {32, 64, 128, 256});
    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::ModuleList downs;
    torch::nn::ModuleList ups;
    DoubleConv bottleneck = nullptr;
    torch::nn::Conv2d final_conv = nullptr;
    torch::nn::MaxPool2d pool = nullptr;
};

TORCH_MODULE(UNet);

} // namespace kvald
