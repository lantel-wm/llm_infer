#include "status.hpp"

namespace core {

Status::Status(int code, std::string err_message)
    : m_code(code), m_message(std::move(err_message)) {}

Status& Status::operator=(int code) {
  m_code = code;
  return *this;
};

bool Status::operator==(int code) const {
  if (m_code == code) {
    return true;
  } else {
    return false;
  }
};

bool Status::operator!=(int code) const {
  if (m_code != code) {
    return true;
  } else {
    return false;
  }
};

Status::operator int() const { return m_code; }

Status::operator bool() const { return m_code == Success; }

int32_t Status::get_err_code() const { return m_code; }

const std::string& Status::get_err_msg() const { return m_message; }

void Status::set_err_msg(const std::string& err_msg) { m_message = err_msg; }

namespace error {

Status Success(std::string err_msg) { return Status{StatusCode::Success, std::move(err_msg)}; }

Status FunctionNotImplement(std::string err_msg) {
  return Status{StatusCode::FunctionUnImplement, std::move(err_msg)};
}

Status PathNotValid(std::string err_msg) {
  return Status{StatusCode::PathNotValid, std::move(err_msg)};
}

Status ModelParseError(std::string err_msg) {
  return Status{StatusCode::ModelParseError, std::move(err_msg)};
}

Status InternalError(std::string err_msg) {
  return Status{StatusCode::InternalError, std::move(err_msg)};
}

Status InvalidArgument(std::string err_msg) {
  return Status{StatusCode::InvalidArgument, std::move(err_msg)};
}

Status KeyHasExits(std::string err_msg) {
  return Status{StatusCode::KeyValueHasExist, std::move(err_msg)};
}
}  // namespace error

std::ostream& operator<<(std::ostream& os, const Status& x) {
  os << x.get_err_msg();
  return os;
}
}  // namespace core