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
        } else if (word == "module") {
          ++moduleDepth;
        } else if (word == "endmodule") {
          if (moduleDepth > 0) {
            --moduleDepth;
          }
        }
        continue;
      }

      // Track operators and other tokens
      lastToken = std::string(1, input[i]);
      ++i;
    }

    // Input is incomplete if:
    // - Unbalanced parentheses
    // - if without matching else
    // - module without matching endmodule
    // - Ends with 'in' keyword (let expression needs body)
    // - Ends with 'then' keyword (if expression needs else)
    // - Ends with binary operator (expression continues)
    return parenDepth > 0 || ifWithoutElse > 0 || moduleDepth > 0 ||
           lastToken == "in" || lastToken == "then" || lastToken == "+" ||
           lastToken == "-" || lastToken == "*" || lastToken == "/" ||
           lastToken == "=" || lastToken == "," || lastToken == "and";
  }
};

#endif // POLANG_INPUT_CHECKER_HPP
