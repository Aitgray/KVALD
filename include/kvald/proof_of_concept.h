#pragma once

#include <string>
#include <stdexcept>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include "kvald/model.h"
#include "kvald/kalman_filter.h"
#include <json.hpp>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>

namespace kvald
{

// --- Custom Exceptions ---
class VideoProcessingError : public std::runtime_error
{
public:
    explicit VideoProcessingError(const std::string& message) : std::runtime_error(message) {}
};

class VideoCaptureError : public VideoProcessingError
{
public:
    explicit VideoCaptureError(const std::string& message) : VideoProcessingError(message) {}
};

class FrameProcessingError : public VideoProcessingError
{
public:
    explicit FrameProcessingError(const std::string& message) : VideoProcessingError(message) {}
};

class MaskVerificationError : public FrameProcessingError
{
public:
    explicit MaskVerificationError(const std::string& message) : FrameProcessingError(message) {}
};

// --- Threading Functions ---
void frame_reader(
    const std::string& video_path,
    std::queue<cv::Mat>& frame_queue,
    std::mutex& queue_mutex,
    std::condition_variable& queue_cond,
    std::atomic<bool>& stop_reading
);

// --- Core Processing Functions ---
void video_processing(
    const std::string& video_path,
    const std::string& output_id,
    kvald::UNet& model,
    const nlohmann::json& config
);

cv::Mat generate_mask(
    kvald::UNet& model,
    const cv::Mat& frame,
    torch::Device device
);

cv::Mat smooth_output(
    const cv::Mat& mask,
    const std::vector<int>& ksize,
    double sigma
);

void verify_feedback(
    const cv::Mat& mask,
    float min_threshold = 0.01f,
    float max_threshold = 0.5f
);

cv::Mat finalize_output(
    const cv::Mat& frame,
    const cv::Mat& mask,
    const cv::Scalar& color = cv::Scalar(0, 0, 255),
    double alpha = 0.4
);

cv::VideoWriter create_video_writer(
    const std::string& filename,
    double fps,
    int width,
    int height,
    bool is_color
);

} // namespace kvald
