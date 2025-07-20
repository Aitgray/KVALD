#include "kvald/model.h"

namespace kvald
{

DoubleConvImpl::DoubleConvImpl(int64_t in_channels, int64_t out_channels)
{
    double_conv = torch::nn::Sequential(
        torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 3).padding(1).bias(false)),
        torch::nn::BatchNorm2d(out_channels),
        torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(out_channels, out_channels, 3).padding(1).bias(false)),
        torch::nn::BatchNorm2d(out_channels),
        torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true))
    );
    register_module("double_conv", double_conv);
}

torch::Tensor DoubleConvImpl::forward(torch::Tensor x)
{
    return double_conv->forward(x);
}

UNetImpl::UNetImpl(int64_t in_channels, int64_t out_channels, const std::vector<int64_t>& features)
{
    // Encoder path
    for (int64_t feature : features)
    {
        downs->push_back(DoubleConv(in_channels, feature));
        in_channels = feature;
    }
    register_module("downs", downs);

    // Bottleneck
    bottleneck = DoubleConv(features.back(), features.back() * 2);
    register_module("bottleneck", bottleneck);

    // Decoder path
    for (long i = features.size() - 1; i >= 0; --i)
    {
        int64_t feature = features[i];
        ups->push_back(torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(feature * 2, feature, 2).stride(2)));
        ups->push_back(DoubleConv(feature * 2, feature));
    }
    register_module("ups", ups);

    // Final conv
    final_conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(features[0], out_channels, 1));
    register_module("final_conv", final_conv);

    // Pooling
    pool = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2));
    register_module("pool", pool);
}

torch::Tensor UNetImpl::forward(torch::Tensor x)
{
    std::vector<torch::Tensor> skip_connections;

    // Encoder
    for (const auto& down : *downs)
    {
        x = down.as<DoubleConv>()->forward(x);
        skip_connections.push_back(x);
        x = pool->forward(x);
    }

    // Bottleneck
    x = bottleneck->forward(x);

    // Decoder
    std::reverse(skip_connections.begin(), skip_connections.end());
    for (size_t i = 0; i < ups->size(); i += 2)
    {
        x = ups[i].as<torch::nn::ConvTranspose2d>()->forward(x);
        torch::Tensor skip = skip_connections[i / 2];

        // pad in case of odd input dimensions
        if (x.sizes() != skip.sizes())
        {
            std::vector<int64_t> pad_dims;
            pad_dims.push_back(0);
            pad_dims.push_back(skip.size(3) - x.size(3));
            pad_dims.push_back(0);
            pad_dims.push_back(skip.size(2) - x.size(2));
            x = torch::nn::functional::pad(x, torch::nn::functional::PadFuncOptions(pad_dims));
        }
        x = torch::cat({skip, x}, 1);
        x = ups[i + 1].as<DoubleConv>()->forward(x);
    }

    return torch::sigmoid(final_conv->forward(x));
}

} // namespace kvald
