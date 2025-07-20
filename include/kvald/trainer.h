#pragma once

#include <torch/torch.h>
#include "kvald/model.h"
#include "kvald/data_loader.h"
#include <json.hpp>

namespace kvald
{

// Loss functions
torch::Tensor supervised_evaluate(torch::Tensor pred_mask, torch::Tensor true_mask);
torch::Tensor unsupervised_evaluate(torch::Tensor pred_mask, torch::Tensor video);

// Metrics
torch::Tensor iou_score(torch::Tensor pred, torch::Tensor target);
torch::Tensor dice_score(torch::Tensor pred, torch::Tensor target);

void train_model(
    std::shared_ptr<GlareDataset>& dataset,
    int epochs,
    int batch_size,
    float lr,
    torch::Device device,
    float loss_weight
);

} // namespace kvald
