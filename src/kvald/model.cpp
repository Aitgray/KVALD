#include "kvald/model.h"
#include <torch/torch.h>
#include <algorithm>
#include <memory>

namespace kvald {

DoubleConvImpl::DoubleConvImpl(int64_t in_channels, int64_t out_channels) {
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

torch::Tensor DoubleConvImpl::forward(torch::Tensor x) {
    return double_conv->forward(x);
}

UNetImpl::UNetImpl(int64_t in_channels, int64_t out_channels, const std::vector<int64_t>& features) {
    // Encoder
    for (int64_t f : features) {
        downs->push_back(DoubleConv(in_channels, f));  // ModuleHolder<DoubleConvImpl>
        in_channels = f;
    }
    register_module("downs", downs);

    // Bottleneck
    bottleneck = DoubleConv(features.back(), features.back() * 2);
    register_module("bottleneck", bottleneck);

    // Decoder (ConvTranspose2d + DoubleConv for each step)
    for (long i = (long)features.size() - 1; i >= 0; --i) {
        int64_t f = features[i];
        ups->push_back(torch::nn::ConvTranspose2d(
            torch::nn::ConvTranspose2dOptions(f * 2, f, 2).stride(2)));
        ups->push_back(DoubleConv(f * 2, f));
    }
    register_module("ups", ups);

    // Final conv
    final_conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(features.front(), out_channels, 1));
    register_module("final_conv", final_conv);

    // Pooling
    pool = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2));
    register_module("pool", pool);
}

torch::Tensor UNetImpl::forward(torch::Tensor x) {
    std::vector<torch::Tensor> skip_connections;
    skip_connections.reserve(downs->size());

    // Encoder path
    for (const auto& m : *downs) {
        // m is shared_ptr<Module>
        auto dbl = std::dynamic_pointer_cast<DoubleConvImpl>(m);
        x = dbl->forward(x);
        skip_connections.push_back(x);
        x = pool->forward(x);
    }

    // Bottleneck
    x = bottleneck->forward(x);

    // Decoder path
    std::reverse(skip_connections.begin(), skip_connections.end());
    for (size_t i = 0; i < ups->size(); i += 2) {
        // ups[i]  : ConvTranspose2d
        // ups[i+1]: DoubleConv
        auto deconv = std::dynamic_pointer_cast<torch::nn::ConvTranspose2dImpl>((*ups)[i]);
        x = deconv->forward(x);

        torch::Tensor skip = skip_connections[i / 2];

        // Pad if shapes differ (odd dims)
        if (x.sizes() != skip.sizes()) {
            std::vector<int64_t> pad = {
                0, skip.size(3) - x.size(3), // W
                0, skip.size(2) - x.size(2)  // H
            };
            x = torch::nn::functional::pad(x, torch::nn::functional::PadFuncOptions(pad));
        }

        x = torch::cat({skip, x}, 1);

        auto dbl = std::dynamic_pointer_cast<DoubleConvImpl>((*ups)[i + 1]);
        x = dbl->forward(x);
    }

    return torch::sigmoid(final_conv->forward(x));
}

} // namespace kvald
