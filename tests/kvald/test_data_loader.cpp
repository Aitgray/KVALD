#include "kvald/data_loader.h"
#include <gtest/gtest.h>

TEST(DataLoaderTest, LoadSyntheticData) {
    kvald::SyntheticData data = kvald::load_synthetic_data("C:/Users/aidan/Desktop/Projects/KVALD/data/synthetic_data.json");
    ASSERT_EQ(data.style, "pulsing");
    ASSERT_EQ(data.video.size(), 10);
    ASSERT_EQ(data.mask.size(), 10);
}
