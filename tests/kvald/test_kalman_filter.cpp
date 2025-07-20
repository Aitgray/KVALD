#include "kvald/kalman_filter.h"
#include <gtest/gtest.h>

TEST(MaskKalmanFilterTest, Update)
{
    Eigen::Array2i shape(1, 1);
    float q_val = 0.1f;
    float r_val = 0.01f;
    kvald::MaskKalmanFilter kf(shape, q_val, r_val);

    Eigen::ArrayXf measurement = Eigen::ArrayXf::Constant(1, 0.5f);
    Eigen::ArrayXf state = kf.update(measurement);

    // Basic check that the state is updated
    ASSERT_NEAR(state(0), 0.4950495f, 1e-6);
}
