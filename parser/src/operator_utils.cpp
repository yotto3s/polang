// clang-format off
#include "parser/operator_utils.hpp"
#include "parser/node.hpp"
#include "parser.hpp" // Must be after node.hpp for bison token types
// clang-format on

namespace polang {

OperatorCategory getOperatorCategory(int op) noexcept {
  switch (op) {
  case yy::parser::token::TPLUS:
  case yy::parser::token::TMINUS:
  case yy::parser::token::TMUL:
  case yy::parser::token::TDIV:
    return OperatorCategory::Arithmetic;
  case yy::parser::token::TCEQ:
  case yy::parser::token::TCNE:
  case yy::parser::token::TCLT:
  case yy::parser::token::TCLE:
  case yy::parser::token::TCGT:
  case yy::parser::token::TCGE:
    return OperatorCategory::Comparison;
  default:
    return OperatorCategory::Unknown;
  }
}

std::string operatorToString(int op) noexcept {
  switch (op) {
  case yy::parser::token::TPLUS:
    return "+";
  case yy::parser::token::TMINUS:
    return "-";
  case yy::parser::token::TMUL:
    return "*";
  case yy::parser::token::TDIV:
    return "/";
  case yy::parser::token::TCEQ:
    return "==";
  case yy::parser::token::TCNE:
    return "!=";
  case yy::parser::token::TCLT:
    return "<";
  case yy::parser::token::TCLE:
    return "<=";
  case yy::parser::token::TCGT:
    return ">";
  case yy::parser::token::TCGE:
    return ">=";
  default:
    return "?";
  }
}

bool isArithmeticOperator(int op) noexcept {
  return getOperatorCategory(op) == OperatorCategory::Arithmetic;
}

bool isComparisonOperator(int op) noexcept {
  return getOperatorCategory(op) == OperatorCategory::Comparison;
}

} // namespace polang
