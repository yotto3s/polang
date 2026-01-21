// clang-format off
#include "parser/operator_utils.hpp"
#include "parser/node.hpp"
#include "parser.hpp" // Must be after node.hpp for bison token types
// clang-format on

namespace polang {

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
  return op == yy::parser::token::TPLUS || op == yy::parser::token::TMINUS ||
         op == yy::parser::token::TMUL || op == yy::parser::token::TDIV;
}

bool isComparisonOperator(int op) noexcept {
  return op == yy::parser::token::TCEQ || op == yy::parser::token::TCNE ||
         op == yy::parser::token::TCLT || op == yy::parser::token::TCLE ||
         op == yy::parser::token::TCGT || op == yy::parser::token::TCGE;
}

} // namespace polang
