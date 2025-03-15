#ifndef STATUS_HPP
#define STATUS_HPP

#include <cstdint>
#include <string>

namespace core {

enum StatusCode : uint8_t {
  Success = 0,
  FunctionUnImplement = 1,
  PathNotValid = 2,
  ModelParseError = 3,
  InternalError = 5,
  KeyValueHasExist = 6,
  InvalidArgument = 7,
};

class Status {
 public:
  Status(int code = StatusCode::Success, std::string err_message = "");

  Status(const Status& other) = default;

  Status& operator=(const Status& other) = default;

  Status& operator=(int code);

  bool operator==(int code) const;

  bool operator!=(int code) const;

  operator int() const;

  operator bool() const;

  int32_t get_err_code() const;

  const std::string& get_err_msg() const;

  void set_err_msg(const std::string& err_msg);

 private:
  int m_code = StatusCode::Success;
  std::string m_message;
};

namespace error {
#define STATUS_CHECK(call)                                                                 \
  do {                                                                                     \
    const base::Status& status = call;                                                     \
    if (!status) {                                                                         \
      const size_t buf_size = 512;                                                         \
      char buf[buf_size];                                                                  \
      snprintf(buf, buf_size - 1,                                                          \
               "Infer error\n File:%s Line:%d\n Error code:%d\n Error msg:%s\n", __FILE__, \
               __LINE__, int(status), status.get_err_msg().c_str());                       \
      LOG(FATAL) << buf;                                                                   \
    }                                                                                      \
  } while (0)

Status Success(std::string err_msg = "");

Status FunctionNotImplement(std::string err_msg = "");

Status PathNotValid(std::string err_msg = "");

Status ModelParseError(std::string err_msg = "");

Status InternalError(std::string err_msg = "");

Status KeyHasExits(std::string err_msg = "");

Status InvalidArgument(std::string err_msg = "");

}  // namespace error

std::ostream& operator<<(std::ostream& os, const Status& x);

}  // namespace core

#endif  // STATUS_HPP