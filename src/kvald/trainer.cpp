#include "kvald/trainer.h"
#include <iostream>
#include <numeric>

namespace kvald
{

torch::Tensor supervised_evaluate(torch::Tensor pred_mask, torch::Tensor true_mask)
{
    return torch::mean(torch::pow(pred_mask - true_mask, 2));
}

torch::Tensor unsupervised_evaluate(torch::Tensor pred_mask, torch::Tensor video)
{
    torch::Tensor avg_before = video.mean();
    torch::Tensor std_before = video.std();
    torch::Tensor masked_video = video * (1 - pred_mask);
    torch::Tensor avg_after = masked_video.mean();
    torch::Tensor std_after = masked_video.std();

    torch::Tensor loss_avg = torch::pow(avg_after - avg_before, 2);
    torch::Tensor loss_std = torch::relu(std_after - std_before);
    return loss_avg + loss_std;
}

torch::Tensor iou_score(torch::Tensor pred, torch::Tensor target)
{
    torch::Tensor intersection = (pred * target).sum();
    torch::Tensor union_val = pred.sum() + target.sum() - intersection;
    return (intersection + 1e-6) / (union_val + 1e-6);
}

torch::Tensor dice_score(torch::Tensor pred, torch::Tensor target)
{
    torch::Tensor intersection = (pred * target).sum();
    return (2. * intersection + 1e-6) / (pred.sum() + target.sum() + 1e-6);
}

void train_model(
    std::shared_ptr<GlareDataset>& dataset,
    int epochs,
    int batch_size,
    float lr,
    torch::Device device,
    float loss_weight
)
{
    // Split into train/val
    size_t n_val = static_cast<size_t>(dataset->size().value() * 0.2);
    size_t n_train = dataset->size().value() - n_val;

    auto datasets = dataset->random_split(n_train, n_val);
    auto train_ds = datasets[0];
    auto val_ds = datasets[1];

    auto train_loader = torch::data::make_data_loader(
        std::move(train_ds), 
        torch::data::DataLoaderOptions().batch_size(batch_size).shuffle(true).workers(4)
    );
    auto val_loader = torch::data::make_data_loader(
        std::move(val_ds), 
        torch::data::DataLoaderOptions().batch_size(batch_size).shuffle(false).workers(4)
    );

    // Model, loss, optimizer
    kvald::UNet model;
    model->to(device);
    torch::nn::BCELoss criterion;
    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(lr));

    for (int epoch = 1; epoch <= epochs; ++epoch)
    {
        // --- Training ---
        model->train();
        double total_loss = 0.0;
        double total_iou = 0.0;
        double total_dice = 0.0;

        for (torch::data::Example<>& batch : *train_loader)
        {
            torch::Tensor frames = batch.data.to(device);
            torch::Tensor masks = batch.target.to(device).to(torch::kFloat32);

            optimizer.zero_grad();
            torch::Tensor preds = model->forward(frames);
            torch::Tensor loss_sup = criterion(preds, masks);
            torch::Tensor loss_unsup = unsupervised_evaluate(preds, frames);
            torch::Tensor loss = loss_sup + loss_weight * loss_unsup;

            loss.backward();
            optimizer.step();

            total_loss += loss.item<double>() * frames.size(0);
            total_iou += iou_score(preds, masks).item<double>() * frames.size(0);
            total_dice += dice_score(preds, masks).item<double>() * frames.size(0);
        }

        size_t n_train_samples = train_loader->dataset().size().value();
        std::cout << "Epoch " << epoch << "/" << epochs << " [Train] ➤ Loss: " << total_loss / n_train_samples
                  << ", IoU: " << total_iou / n_train_samples << ", Dice: " << total_dice / n_train_samples << std::endl;

        // --- Validation ---
        model->eval();
        double val_loss = 0.0;
        double val_iou = 0.0;
        double val_dice = 0.0;

        torch::NoGradGuard no_grad;
        for (torch::data::Example<>& batch : *val_loader)
        {
            torch::Tensor frames = batch.data.to(device);
            torch::Tensor masks = batch.target.to(device).to(torch::kFloat32);

            torch::Tensor preds = model->forward(frames);
            torch::Tensor loss_sup = criterion(preds, masks);

            val_loss += loss_sup.item<double>() * frames.size(0);
            val_iou += iou_score((preds > 0.5).to(torch::kInt32), masks.to(torch::kInt32)).item<double>() * frames.size(0);
            val_dice += dice_score((preds > 0.5).to(torch::kInt32), masks.to(torch::kInt32)).item<double>() * frames.size(0);
        }

        size_t n_val_samples = val_loader->dataset().size().value();
        std::cout << " Val ➤ Loss: " << val_loss / n_val_samples
                  << ", IoU: " << val_iou / n_val_samples << ", Dice: " << val_dice / n_val_samples << std::endl;
    }

    // Save checkpoint
    torch::save(model, "kvald_unet.pt");
    std::cout << "Training complete. Model saved to kvald_unet.pt" << std::endl;
}

} // namespace kvald
