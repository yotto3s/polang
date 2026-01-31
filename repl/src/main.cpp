#include "repl/repl_session.hpp"

#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <variant>

// Cross-platform terminal detection
#ifdef _WIN32
#include <io.h>
#define isatty _isatty
#define fileno _fileno
#else
#include <unistd.h>
#endif

// Print the result of an evaluation
static void printResult(const EvalResult& result) noexcept {
  if (!result.success || !result.hasValue()) {
    return;
  }

  try {
    std::visit(
        [&](auto&& val) {
          using T = std::decay_t<decltype(val)>;
          if constexpr (std::is_same_v<T, std::monostate>) {
            // No value
          } else if constexpr (std::is_same_v<T, bool>) {
            std::cout << (val ? "true" : "false") << " : bool\n";
          } else if constexpr (std::is_same_v<T, double>) {
            std::cout << val << " : " << result.type << "\n";
          } else if constexpr (std::is_same_v<T, float>) {
            std::cout << static_cast<double>(val) << " : " << result.type
                      << "\n";
          } else {
            // int64_t, int8_t, int16_t — all printed as integer
            std::cout << static_cast<int64_t>(val) << " : " << result.type
                      << "\n";
          }
        },
        result.value);
  } catch (...) {
    // unreachable
    assert(false);
  }
}

// Read input with multi-line support for incomplete expressions
static std::string readInput(std::istream& in, bool interactive) {
  std::string buffer;
  std::string line;

  // Show initial prompt for interactive mode
  if (interactive) {
    std::cout << "> " << std::flush;
  }

  while (std::getline(in, line)) {
    if (!buffer.empty()) {
      buffer += "\n";
    }
    buffer += line;

    // Check if input is complete
    if (!ReplSession::isInputIncomplete(buffer)) {
      break;
    }

    // Show continuation prompt for interactive mode
    if (interactive) {
      std::cout << "... " << std::flush;
    }
  }

  return buffer;
}

int main(int argc, char** argv) {
  ReplSession session;

  if (!session.initialize()) {
    std::cerr << "Failed to initialize REPL session\n";
    return 1;
  }

  // File mode: read file, evaluate line by line, and exit
  if (argc > 1) {
    std::ifstream file(argv[1]);
    if (!file.is_open()) {
      std::cerr << "Error: Cannot open file: " << argv[1] << "\n";
      return 1;
    }

    // Evaluate each line separately to avoid parser ambiguity
    // (e.g., `10\n*x` should be two statements, not `10 * x`)
    EvalResult lastResult = EvalResult::ok();
    while (true) {
      const std::string input = readInput(file, false);
      if (file.eof() && input.empty()) {
        break;
      }
      if (input.empty()) {
        continue;
      }

      lastResult = session.evaluate(input);
      if (!lastResult.success) {
        return 1;
      }
    }

    printResult(lastResult);
    return 0;
  }

  // Determine if running interactively (stdin is a terminal)
  const bool interactive = isatty(fileno(stdin)) != 0;

  if (interactive) {
    std::cout << "Polang REPL (type 'exit' or Ctrl+D to quit)\n";
  }

  while (true) {
    const std::string input = readInput(std::cin, interactive);

    // Check for EOF
    if (std::cin.eof() && input.empty()) {
      break;
    }

    // Check for exit command
    if (input == "exit" || input == "quit") {
      break;
    }

    // Skip empty input
    if (input.empty()) {
      continue;
    }

    // Evaluate the input
    const EvalResult result = session.evaluate(input);
    printResult(result);

    // In non-interactive mode, exit on first error
    if (!result.success && !interactive) {
      return 1;
    }
  }

  if (interactive) {
    std::cout << "\n";
  }

  return 0;
}
