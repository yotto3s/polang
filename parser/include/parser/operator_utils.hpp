#ifndef POLANG_OPERATOR_UTILS_HPP
#define POLANG_OPERATOR_UTILS_HPP

#include <string>

namespace polang {

/// Convert a binary operator token to its string representation.
/// @param op The operator token (e.g., TPLUS, TMINUS, etc.)
/// @return The string representation (e.g., "+", "-", etc.) or "?" for unknown
std::string operatorToString(int op) noexcept;

/// Check if the given token represents an arithmetic operator.
/// Arithmetic operators: +, -, *, /
/// @param op The operator token
/// @return true if the operator is arithmetic
bool isArithmeticOperator(int op) noexcept;

/// Check if the given token represents a comparison operator.
/// Comparison operators: ==, !=, <, <=, >, >=
/// @param op The operator token
/// @return true if the operator is a comparison
bool isComparisonOperator(int op) noexcept;

} // namespace polang

#endif // POLANG_OPERATOR_UTILS_HPP
