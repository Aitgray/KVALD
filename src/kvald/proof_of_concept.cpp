#include "kvald/proof_of_concept.h"
#include <iostream>
#include <iomanip>

namespace kvald
{

// --- Threading Functions ---
void frame_reader(
    const std::string& video_path,
    std::queue<cv::Mat>& frame_queue,
    std::mutex& queue_mutex,
    std::condition_variable& queue_cond,
    std::atomic<bool>& stop_reading
)
{
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened())
    {
        stop_reading = true;
        std::lock_guard<std::mutex> lock(queue_mutex);
        frame_queue.push(cv::Mat()); // Sentinel for error/end
        queue_cond.notify_one();
        throw VideoCaptureError("Cannot open video file: " + video_path);
    }

    while (!stop_reading)
    {
        cv::Mat frame;
        if (!cap.read(frame))
        {
            break; // End of video
        }

        std::lock_guard<std::mutex> lock(queue_mutex);
        frame_queue.push(frame);
        queue_cond.notify_one();
    }

    cap.release();
    std::lock_guard<std::mutex> lock(queue_mutex);
    frame_queue.push(cv::Mat()); // Sentinel to indicate the end of the stream
    queue_cond.notify_one();
}

// --- Core Processing Functions ---
cv::Mat generate_mask(
    kvald::UNet& model,
    const cv::Mat& frame,
    torch::Device device
)
{
    // Convert cv::Mat to torch::Tensor
    // OpenCV uses HWC, PyTorch expects NCHW
    torch::Tensor inp = torch::from_blob(
        frame.data,
        {1, frame.rows, frame.cols},
        torch::TensorOptions().dtype(torch::kFloat32)
    );
    inp = inp.to(device);

    torch::Tensor out;
    {
        torch::NoGradGuard no_grad;
        out = model->forward(inp.unsqueeze(0)); // Add batch dimension
    }

    // Convert torch::Tensor back to cv::Mat
    out = out.squeeze().cpu(); // Remove batch and channel dimensions, move to CPU
    cv::Mat mask(out.sizes()[0], out.sizes()[1], CV_32F, out.data_ptr<float>());
    return mask.clone(); // Clone to ensure data ownership
}

cv::Mat smooth_output(
    const cv::Mat& mask,
    const std::vector<int>& ksize,
    double sigma
)
{
    cv::Mat smoothed_mask;
    cv::GaussianBlur(mask, smoothed_mask, cv::Size(ksize[0], ksize[1]), sigma);
    return smoothed_mask;
}

void verify_feedback(
    const cv::Mat& mask,
    float min_threshold,
    float max_threshold
)
{
    double mask_proportion = cv::mean(mask)[0];
    if (!(mask_proportion > min_threshold && mask_proportion < max_threshold))
    {
        std::stringstream ss;
        ss << "Mask covers " << std::fixed << std::setprecision(2) << mask_proportion * 100 << "% of the frame, "
           << "which is outside the acceptable range of (" << min_threshold * 100 << "%, " << max_threshold * 100 << "%).";
        throw MaskVerificationError(ss.str());
    }
}

cv::Mat finalize_output(
    const cv::Mat& frame,
    const cv::Mat& mask,
    const cv::Scalar& color,
    double alpha
)
{
    cv::Mat overlay = frame.clone();
    cv::Mat colored_mask;
    cv::cvtColor(mask, colored_mask, cv::COLOR_GRAY2BGR);
    colored_mask.convertTo(colored_mask, CV_8UC3, 255.0);
    colored_mask = colored_mask.mul(color, 1.0/255.0);

    cv::addWeighted(colored_mask, alpha, overlay, 1 - alpha, 0, overlay);
    return overlay;
}

cv::VideoWriter create_video_writer(
    const std::string& filename,
    double fps,
    int width,
    int height,
    bool is_color
)
{
    int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
    cv::VideoWriter writer(filename, fourcc, fps, cv::Size(width, height), is_color);
    if (!writer.isOpened())
    {
        throw std::ios_base::failure("Failed to open VideoWriter for " + filename);
    }
    return writer;
}

void video_processing(
    const std::string& video_path,
    const std::string& output_id,
    kvald::UNet& model,
    const nlohmann::json& config
)
{
    // Load configuration
    std::vector<int> smoothing_ksize = config["smoothing"]["kernel_size"].get<std::vector<int>>();
    double smoothing_sigma = config["smoothing"]["sigma"];
    std::string device_str = config["train"]["device"];
    torch::Device device(device_str);

    model->to(device);
    model->eval();

    // Get video properties for output
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened())
    {
        throw VideoCaptureError("Cannot open video file: " + video_path);
    }
    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    cap.release();

    // Initialize video writers
    cv::VideoWriter mask_writer = create_video_writer(output_id + "_mask.mp4", fps, width, height, false);
    cv::VideoWriter final_writer = create_video_writer(output_id + "_final.mp4", fps, width, height, true);

    // Initialize Kalman Filter
    kvald::MaskKalmanFilter kf(
        Eigen::Array2i(height, width),
        config["kalman_filter"]["Q"],
        config["kalman_filter"]["R"]
    );

    // Frame queue and threading events
    std::queue<cv::Mat> frame_queue;
    std::mutex queue_mutex;
    std::condition_variable queue_cond;
    std::atomic<bool> stop_reading(false);

    // Start the frame reader thread
    std::thread reader_thread(frame_reader, video_path, std::ref(frame_queue), std::ref(queue_mutex), std::ref(queue_cond), std::ref(stop_reading));

    int frame_count = 0;
    try
    {
        while (true)
        {
            cv::Mat frame;
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                queue_cond.wait(lock, [&]{ return !frame_queue.empty(); });
                frame = frame_queue.front();
                frame_queue.pop();
            }

            if (frame.empty()) // Sentinel for end of stream or error
            {
                if (stop_reading && frame_queue.empty()) // If reading stopped due to error and queue is empty
                {
                    break;
                }
                else if (!stop_reading) // Normal end of video
                {
                    break;
                }
                else // Error during reading, but queue might still have frames
                {
                    // This case should ideally be handled by throwing from frame_reader
                    // but as a fallback, we break if we get an empty frame and stop_reading is true
                    break;
                }
            }

            frame_count++;
            cv::Mat gray_frame;
            cv::cvtColor(frame, gray_frame, cv::COLOR_BGR2GRAY);
            gray_frame.convertTo(gray_frame, CV_32F, 1.0 / 255.0);

            cv::Mat mask = generate_mask(model, gray_frame, device);
            mask = smooth_output(mask, smoothing_ksize, smoothing_sigma);
            
            // Convert cv::Mat to Eigen::ArrayXf for Kalman Filter
            Eigen::Map<Eigen::MatrixXf> eigen_mask_map(reinterpret_cast<float*>(mask.data), mask.rows, mask.cols);
            Eigen::ArrayXf filtered_mask_eigen = kf.update(Eigen::Map<Eigen::ArrayXf>(eigen_mask_map.data(), eigen_mask_map.size()));

            // Convert Eigen::ArrayXf back to cv::Mat
            cv::Mat filtered_mask_cv(mask.rows, mask.cols, CV_32F);
            Eigen::Map<Eigen::ArrayXf>(reinterpret_cast<float*>(filtered_mask_cv.data), filtered_mask_eigen.size()) = filtered_mask_eigen;

            try
            {
                verify_feedback(filtered_mask_cv);
            }
            catch (const MaskVerificationError& e)
            {
                std::cerr << "Warning: Frame " << frame_count << " failed verification: " << e.what() << std::endl;
                continue;
            }

            cv::Mat mask_output;
            filtered_mask_cv.convertTo(mask_output, CV_8U, 255.0);
            
            cv::Mat final_output = finalize_output(frame, filtered_mask_cv);

            mask_writer.write(mask_output);
            final_writer.write(final_output);
        }
    }
    catch (const VideoProcessingError& e)
    {
        std::cerr << "An error occurred during video processing: " << e.what() << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cerr << "An unexpected error occurred: " << e.what() << std::endl;
    }

    // Clean up
    stop_reading = true;
    queue_cond.notify_one(); // Notify reader to unblock if waiting
    if (reader_thread.joinable())
    {
        reader_thread.join();
    }
    mask_writer.release();
    final_writer.release();

    if (frame_count == 0)
    {
        throw FrameProcessingError("No frames were read from the video.");
    }

    std::cout << "Processing complete. Videos saved as " << output_id << "_mask.mp4 and " << output_id << "_final.mp4" << std::endl;
}

} // namespace kvald
