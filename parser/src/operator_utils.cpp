// clang-format off
#include "parser/operator_utils.hpp"
#include "parser/node.hpp"
#include "parser.hpp" // Must be after node.hpp for bison union types
// clang-format on

namespace polang {

std::string operatorToString(int op) noexcept {
  switch (op) {
  case TPLUS:
    return "+";
  case TMINUS:
    return "-";
  case TMUL:
    return "*";
  case TDIV:
    return "/";
  case TCEQ:
    return "==";
  case TCNE:
    return "!=";
  case TCLT:
    return "<";
  case TCLE:
    return "<=";
  case TCGT:
    return ">";
  case TCGE:
    return ">=";
  default:
    return "?";
  }
}

bool isArithmeticOperator(int op) noexcept {
  return op == TPLUS || op == TMINUS || op == TMUL || op == TDIV;
}

bool isComparisonOperator(int op) noexcept {
  return op == TCEQ || op == TCNE || op == TCLT || op == TCLE || op == TCGT ||
         op == TCGE;
}

} // namespace polang
