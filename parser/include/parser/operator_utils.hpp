#ifndef POLANG_OPERATOR_UTILS_HPP
#define POLANG_OPERATOR_UTILS_HPP

#include <string>

namespace polang {

/// Categories of binary operators.
enum class OperatorCategory {
  Arithmetic, ///< +, -, *, /
  Comparison, ///< ==, !=, <, <=, >, >=
  Unknown     ///< Unrecognized operator
};

/// Get the category of a binary operator.
/// @param op The operator token
/// @return The category of the operator
[[nodiscard]] OperatorCategory getOperatorCategory(int op) noexcept;

/// Convert a binary operator token to its string representation.
/// @param op The operator token (e.g., TPLUS, TMINUS, etc.)
/// @return The string representation (e.g., "+", "-", etc.) or "?" for unknown
[[nodiscard]] std::string operatorToString(int op) noexcept;

/// Check if the given token represents an arithmetic operator.
/// Arithmetic operators: +, -, *, /
/// @param op The operator token
/// @return true if the operator is arithmetic
[[nodiscard]] bool isArithmeticOperator(int op) noexcept;

/// Check if the given token represents a comparison operator.
/// Comparison operators: ==, !=, <, <=, >, >=
/// @param op The operator token
/// @return true if the operator is a comparison
[[nodiscard]] bool isComparisonOperator(int op) noexcept;

} // namespace polang

#endif // POLANG_OPERATOR_UTILS_HPP
