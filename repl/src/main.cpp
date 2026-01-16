#include "repl/repl_session.hpp"

#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

// Cross-platform terminal detection
#ifdef _WIN32
#include <io.h>
#define isatty _isatty
#define fileno _fileno
#else
#include <unistd.h>
#endif

// Print the result of an evaluation
static void printResult(const EvalResult& result) {
  if (!result.success) {
    // Error already printed to stderr by parser/type checker
    return;
  }

  if (!result.hasValue) {
    // No value to print (void result)
    return;
  }

  if (result.type == "double") {
    double d = 0;
    std::memcpy(&d, &result.rawValue, sizeof(double));
    std::cout << d << " : double\n";
  } else if (result.type == "bool") {
    std::cout << (result.rawValue != 0 ? "true" : "false") << " : bool\n";
  } else if (result.type == "int") {
    std::cout << result.rawValue << " : int\n";
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

  // File mode: read file, evaluate, and exit
  if (argc > 1) {
    std::ifstream file(argv[1]);
    if (!file.is_open()) {
      std::cerr << "Error: Cannot open file: " << argv[1] << "\n";
      return 1;
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    const std::string content = buffer.str();

    const EvalResult result = session.evaluate(content);
    printResult(result);
    return result.success ? 0 : 1;
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
