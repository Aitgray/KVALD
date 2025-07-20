#include "kvald/kalman_filter.h"

namespace kvald
{

MaskKalmanFilter::MaskKalmanFilter(const Eigen::Array2i& shape, float q_val, float r_val)
    : Q(q_val), R(r_val)
{
    state = Eigen::ArrayXf::Zero(shape.prod());
    covariance = Eigen::ArrayXf::Ones(shape.prod());
}

Eigen::ArrayXf MaskKalmanFilter::update(const Eigen::ArrayXf& measurement)
{
    // Prediction step
    Eigen::ArrayXf predicted_state = state;
    Eigen::ArrayXf predicted_covariance = covariance + Q;

    // Update step
    Eigen::ArrayXf innovation = measurement - predicted_state;
    Eigen::ArrayXf innovation_covariance = predicted_covariance + R;
    Eigen::ArrayXf kalman_gain = predicted_covariance / innovation_covariance;

    // Update state and covariance
    state = predicted_state + kalman_gain * innovation;
    covariance = (1 - kalman_gain) * predicted_covariance;

    return state;
}

} // namespace kvald
