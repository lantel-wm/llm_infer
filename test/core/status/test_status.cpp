#include <gtest/gtest.h>
#include "core/status/status.hpp"

namespace core {
namespace {

// Test fixture for Status tests
class StatusTest : public ::testing::Test {
 protected:
  void SetUp() override {}
};

// Test default constructor
TEST_F(StatusTest, DefaultConstructor) {
  Status status;
  EXPECT_EQ(status.get_err_code(), StatusCode::Success);
  EXPECT_EQ(status.get_err_msg(), "");
  EXPECT_TRUE(status);  // Tests bool operator
}

// Test parameterized constructor
TEST_F(StatusTest, ParameterizedConstructor) {
  Status status(StatusCode::InvalidArgument, "Invalid input");
  EXPECT_EQ(status.get_err_code(), StatusCode::InvalidArgument);
  EXPECT_EQ(status.get_err_msg(), "Invalid input");
  EXPECT_FALSE(status);  // Tests bool operator
}

// Test copy constructor
TEST_F(StatusTest, CopyConstructor) {
  Status original(StatusCode::PathNotValid, "Path not found");
  Status copy(original);

  EXPECT_EQ(copy.get_err_code(), original.get_err_code());
  EXPECT_EQ(copy.get_err_msg(), original.get_err_msg());
}

// Test assignment operator (Status)
TEST_F(StatusTest, AssignmentOperator) {
  Status original(StatusCode::ModelParseError, "Parse error");
  Status assigned = original;

  EXPECT_EQ(assigned.get_err_code(), original.get_err_code());
  EXPECT_EQ(assigned.get_err_msg(), original.get_err_msg());
}

// Test assignment operator (int)
TEST_F(StatusTest, IntAssignmentOperator) {
  Status status;
  status = StatusCode::InternalError;

  EXPECT_EQ(status.get_err_code(), StatusCode::InternalError);
}

// Test equality operator
TEST_F(StatusTest, EqualityOperator) {
  Status status(StatusCode::FunctionUnImplement);
  EXPECT_TRUE(status == StatusCode::FunctionUnImplement);
  EXPECT_FALSE(status == StatusCode::Success);
}

// Test inequality operator
TEST_F(StatusTest, InequalityOperator) {
  Status status(StatusCode::KeyValueHasExist);
  EXPECT_TRUE(status != StatusCode::Success);
  EXPECT_FALSE(status != StatusCode::KeyValueHasExist);
}

// Test int conversion operator
TEST_F(StatusTest, IntConversionOperator) {
  Status status(StatusCode::InvalidArgument);
  int code = status;
  EXPECT_EQ(code, StatusCode::InvalidArgument);
}

// Test bool conversion operator
TEST_F(StatusTest, BoolConversionOperator) {
  Status success_status(StatusCode::Success);
  Status error_status(StatusCode::InternalError);

  EXPECT_TRUE(success_status);
  EXPECT_FALSE(error_status);
}

// Test error message manipulation
TEST_F(StatusTest, ErrorMessageManipulation) {
  Status status;
  EXPECT_EQ(status.get_err_msg(), "");

  status.set_err_msg("New error message");
  EXPECT_EQ(status.get_err_msg(), "New error message");
}

// Test error namespace factory functions
TEST_F(StatusTest, ErrorFactoryFunctions) {
  // Test Success
  auto success = error::Success("Operation successful");
  EXPECT_EQ(success.get_err_code(), StatusCode::Success);
  EXPECT_EQ(success.get_err_msg(), "Operation successful");

  // Test FunctionNotImplement
  auto not_implemented = error::FunctionNotImplement("Function not ready");
  EXPECT_EQ(not_implemented.get_err_code(), StatusCode::FunctionUnImplement);
  EXPECT_EQ(not_implemented.get_err_msg(), "Function not ready");

  // Test PathNotValid
  auto path_error = error::PathNotValid("Invalid path");
  EXPECT_EQ(path_error.get_err_code(), StatusCode::PathNotValid);
  EXPECT_EQ(path_error.get_err_msg(), "Invalid path");

  // Test ModelParseError
  auto parse_error = error::ModelParseError("Parse failed");
  EXPECT_EQ(parse_error.get_err_code(), StatusCode::ModelParseError);
  EXPECT_EQ(parse_error.get_err_msg(), "Parse failed");

  // Test InternalError
  auto internal_error = error::InternalError("Internal failure");
  EXPECT_EQ(internal_error.get_err_code(), StatusCode::InternalError);
  EXPECT_EQ(internal_error.get_err_msg(), "Internal failure");

  // Test InvalidArgument
  auto invalid_arg = error::InvalidArgument("Bad argument");
  EXPECT_EQ(invalid_arg.get_err_code(), StatusCode::InvalidArgument);
  EXPECT_EQ(invalid_arg.get_err_msg(), "Bad argument");

  // Test KeyHasExits
  auto key_exists = error::KeyHasExits("Key already exists");
  EXPECT_EQ(key_exists.get_err_code(), StatusCode::KeyValueHasExist);
  EXPECT_EQ(key_exists.get_err_msg(), "Key already exists");
}

// Test stream operator
TEST_F(StatusTest, StreamOperator) {
  Status status(StatusCode::InternalError, "Test error");
  std::ostringstream oss;
  oss << status;
  EXPECT_EQ(oss.str(), "Test error");
}

// Test default error messages
TEST_F(StatusTest, DefaultErrorMessages) {
  EXPECT_EQ(error::Success().get_err_msg(), "");
  EXPECT_EQ(error::FunctionNotImplement().get_err_msg(), "");
  EXPECT_EQ(error::PathNotValid().get_err_msg(), "");
  EXPECT_EQ(error::ModelParseError().get_err_msg(), "");
  EXPECT_EQ(error::InternalError().get_err_msg(), "");
  EXPECT_EQ(error::InvalidArgument().get_err_msg(), "");
  EXPECT_EQ(error::KeyHasExits().get_err_msg(), "");
}

}  // namespace
}  // namespace core

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}