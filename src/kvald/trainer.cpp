#include "kvald/trainer.h"
#include "kvald/data_loader.h"
#include "kvald/model.h"

#include <torch/torch.h>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

namespace kvald {

// ---- Implement the functions declared in trainer.h (exact same signatures) ----
torch::Tensor supervised_evaluate(torch::Tensor pred_mask, torch::Tensor true_mask) {
    return torch::mean(torch::pow(pred_mask - true_mask, 2));
}

torch::Tensor unsupervised_evaluate(torch::Tensor pred_mask, torch::Tensor video) {
    auto avg_before   = video.mean();
    auto std_before   = video.std();
    auto masked_video = video * (1 - pred_mask);
    auto avg_after    = masked_video.mean();
    auto std_after    = masked_video.std();

    auto loss_avg = torch::pow(avg_after - avg_before, 2);
    auto loss_std = torch::relu(std_after - std_before);
    return loss_avg + loss_std;
}

torch::Tensor iou_score(torch::Tensor pred, torch::Tensor target) {
    auto intersection = (pred * target).sum();
    auto union_val    = pred.sum() + target.sum() - intersection;
    return (intersection + 1e-6) / (union_val + 1e-6);
}

torch::Tensor dice_score(torch::Tensor pred, torch::Tensor target) {
    auto intersection = (pred * target).sum();
    return (2.0 * intersection + 1e-6) / (pred.sum() + target.sum() + 1e-6);
}

// ---------------- Simple batching (avoid DataLoader template hell) -------------
struct Batch {
    torch::Tensor data;
    torch::Tensor target;
};

static Batch make_batch(GlareDataset& ds,
                        const std::vector<size_t>& idxs,
                        torch::Device device) {
    std::vector<torch::Tensor> datas, targets;
    datas.reserve(idxs.size());
    targets.reserve(idxs.size());
    for (auto i : idxs) {
        auto ex = ds.get(i);
        datas.push_back(ex.data);
        targets.push_back(ex.target);
    }
    auto data   = torch::cat(datas, 0).to(device);
    auto target = torch::cat(targets, 0).to(device).to(torch::kFloat32);
    return {data, target};
}

// ------------------------------- Training -------------------------------------
void train_model(std::shared_ptr<GlareDataset>& dataset,
                 int epochs,
                 int batch_size,
                 float lr,
                 torch::Device device,
                 float loss_weight)
{
    // Split indices 80/20
    const size_t total   = dataset->size().value();
    const size_t n_val   = static_cast<size_t>(total * 0.2);
    const size_t n_train = total - n_val;

    std::vector<size_t> all_idx(total);
    std::iota(all_idx.begin(), all_idx.end(), 0);

    std::mt19937 rng(std::random_device{}());
    std::shuffle(all_idx.begin(), all_idx.end(), rng);

    std::vector<size_t> train_idx(all_idx.begin(), all_idx.begin() + n_train);
    std::vector<size_t> val_idx  (all_idx.begin() + n_train, all_idx.end());

    UNet model;
    model->to(device);

    torch::nn::BCELoss criterion;
    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(lr));

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        // -------- Train --------
        model->train();
        double tot_loss = 0.0, tot_iou = 0.0, tot_dice = 0.0;
        size_t seen = 0;

        std::shuffle(train_idx.begin(), train_idx.end(), rng);

        for (size_t off = 0; off < n_train; off += static_cast<size_t>(batch_size)) {
            size_t end = std::min(off + static_cast<size_t>(batch_size), n_train);
            std::vector<size_t> slice(train_idx.begin() + off, train_idx.begin() + end);
            auto batch = make_batch(*dataset, slice, device);

            optimizer.zero_grad();
            auto preds      = model->forward(batch.data);
            auto loss_sup   = criterion(preds, batch.target);
            auto loss_unsup = unsupervised_evaluate(preds, batch.data);
            auto loss       = loss_sup + loss_weight * loss_unsup;

            loss.backward();
            optimizer.step();

            size_t bs = slice.size();
            seen     += bs;
            tot_loss += loss.item().toFloat() * bs;
            tot_iou  += iou_score(preds, batch.target).item().toFloat() * bs;
            tot_dice += dice_score(preds, batch.target).item().toFloat() * bs;
        }

        std::cout << "Epoch " << epoch << "/" << epochs
                  << " [Train] ➤ Loss: " << tot_loss / seen
                  << ", IoU: "  << tot_iou  / seen
                  << ", Dice: " << tot_dice / seen << '\n';

        // -------- Val --------
        model->eval();
        double v_loss = 0.0, v_iou = 0.0, v_dice = 0.0;
        size_t vseen = 0;

        torch::NoGradGuard no_grad;
        for (size_t off = 0; off < n_val; off += static_cast<size_t>(batch_size)) {
            size_t end = std::min(off + static_cast<size_t>(batch_size), n_val);
            std::vector<size_t> slice(val_idx.begin() + off, val_idx.begin() + end);
            auto batch = make_batch(*dataset, slice, device);

            auto preds    = model->forward(batch.data);
            auto loss_sup = criterion(preds, batch.target);

            size_t bs = slice.size();
            vseen    += bs;
            v_loss   += loss_sup.item().toFloat() * bs;
            v_iou    += iou_score((preds > 0.5).to(torch::kInt32),
                                  batch.target.to(torch::kInt32)).item().toFloat() * bs;
            v_dice   += dice_score((preds > 0.5).to(torch::kInt32),
                                   batch.target.to(torch::kInt32)).item().toFloat() * bs;
        }

        std::cout << "        [Val]   ➤ Loss: " << v_loss / vseen
                  << ", IoU: "  << v_iou  / vseen
                  << ", Dice: " << v_dice / vseen << '\n';
    }

    torch::save(model, "kvald_unet.pt");
    std::cout << "Training complete. Model saved to kvald_unet.pt\n";
}

} // namespace kvald
