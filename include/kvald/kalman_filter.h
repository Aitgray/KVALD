#pragma once

#include <Eigen/Dense>

namespace kvald
{

class MaskKalmanFilter
{
public:
    MaskKalmanFilter(const Eigen::Array2i& shape, float q_val, float r_val);

    Eigen::ArrayXf update(const Eigen::ArrayXf& measurement);

private:
    float Q;
    float R;
    Eigen::ArrayXf state;
    Eigen::ArrayXf covariance;
};

} // namespace kvald
