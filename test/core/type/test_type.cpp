#include <glog/logging.h>
#include <gtest/gtest.h>
#include "type.hpp"

namespace core {
namespace {

// Test fixture for Status tests
class TypeTest : public ::testing::Test {
 protected:
  void SetUp() override {}
};

// Test default constructor
TEST_F(TypeTest, data_type_ostream) {
  DataType fp32_type = DataType::FP32;
  DataType int8_type = DataType::INT8;
  DataType int32_type = DataType::INT32;
  DataType unknown_type = DataType::Unknown;

  LOG(INFO) << fp32_type;
  LOG(INFO) << int8_type;
  LOG(INFO) << int32_type;
  LOG(INFO) << unknown_type;

  std::stringstream ss;
  ss << fp32_type;
  EXPECT_EQ(ss.str(), "DataType::FP32");

  ss.str("");
  ss << int8_type;
  EXPECT_EQ(ss.str(), "DataType::INT8");

  ss.str("");
  ss << int32_type;
  EXPECT_EQ(ss.str(), "DataType::INT32");

  ss.str("");
  ss << unknown_type;
  EXPECT_EQ(ss.str(), "DataType::Unknown");
}

}  // namespace
}  // namespace core