#ifndef POLANG_INPUT_CHECKER_HPP
#define POLANG_INPUT_CHECKER_HPP

#include <cctype>
#include <string>

// Utility class for checking input completeness
// Separated from ReplSession to allow testing without LLVM dependencies
class InputChecker {
public:
  // Check if input appears incomplete (needs more lines)
  static bool isInputIncomplete(const std::string& input) noexcept {
    int parenDepth = 0;
    int ifWithoutElse = 0;
    int letWithoutIn = 0;
    int moduleDepth = 0;
    std::string lastToken;

    // Simple tokenization - track keywords and brackets
    std::size_t i = 0;
    while (i < input.size()) {
      // Skip whitespace
      if (std::isspace(static_cast<unsigned char>(input[i])) != 0) {
        ++i;
        continue;
      }

      // Skip block comments (* ... *) with nesting
      if (input[i] == '(' && i + 1 < input.size() && input[i + 1] == '*') {
        int depth = 1;
        i += 2;
        while (i < input.size() && depth > 0) {
          if (input[i] == '(' && i + 1 < input.size() && input[i + 1] == '*') {
            ++depth;
            i += 2;
          } else if (input[i] == '*' && i + 1 < input.size() &&
                     input[i + 1] == ')') {
            --depth;
            i += 2;
          } else {
            ++i;
          }
        }
        if (depth > 0) {
          return true; // Unterminated comment - input is incomplete
        }
        continue;
      }

      // Track parentheses
      if (input[i] == '(') {
        ++parenDepth;
        lastToken = "(";
        ++i;
        continue;
      }
      if (input[i] == ')') {
        --parenDepth;
        lastToken = ")";
        ++i;
        continue;
      }

      // Check for numeric literals
      if (std::isdigit(static_cast<unsigned char>(input[i])) != 0) {
        while (i < input.size() &&
               (std::isdigit(static_cast<unsigned char>(input[i])) != 0 ||
                input[i] == '.')) {
          ++i;
        }
        lastToken = "number";
        continue;
      }

      // Check for keywords and identifiers
      if (std::isalpha(static_cast<unsigned char>(input[i])) != 0 ||
          input[i] == '_') {
        const std::size_t start = i;
        while (i < input.size() &&
               (std::isalnum(static_cast<unsigned char>(input[i])) != 0 ||
                input[i] == '_')) {
          ++i;
        }
        const std::string word = input.substr(start, i - start);
        lastToken = word;

        if (word == "if") {
          ++ifWithoutElse;
        } else if (word == "else") {
          if (ifWithoutElse > 0) {
            --ifWithoutElse;
          }
        } else if (word == "let") {
          ++letWithoutIn;
        } else if (word == "in") {
          if (letWithoutIn > 0) {
            --letWithoutIn;
          }
        } else if (word == "module") {
          ++moduleDepth;
        } else if (word == "endmodule") {
          if (moduleDepth > 0) {
            --moduleDepth;
          }
        }
        continue;
      }

      // Track operators and other tokens (including multi-character operators)
      if ((input[i] == '&' || input[i] == '|') && i + 1 < input.size() &&
          input[i + 1] == input[i]) {
        lastToken = std::string(2, input[i]);
        i += 2;
      } else {
        lastToken = std::string(1, input[i]);
        ++i;
      }
    }

    // Input is incomplete if:
    // - Unbalanced parentheses
    // - if without matching else
    // - let without matching in
    // - module without matching endmodule
    // - Ends with 'in' keyword (let expression needs body)
    // - Ends with 'then' keyword (if expression needs else)
    // - Ends with binary operator (expression continues)
    return parenDepth > 0 || ifWithoutElse > 0 || letWithoutIn > 0 ||
           moduleDepth > 0 || lastToken == "in" || lastToken == "then" ||
           lastToken == "+" || lastToken == "-" || lastToken == "*" ||
           lastToken == "/" || lastToken == "%" || lastToken == "=" ||
           lastToken == "," || lastToken == "&&" || lastToken == "||" ||
           lastToken == "and";
  }
};

#endif // POLANG_INPUT_CHECKER_HPP
