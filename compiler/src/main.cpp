#include "compiler/mlir_codegen.hpp"
#include "parser/ast_printer.hpp"
#include "parser/node.hpp"
#include "parser/parser_api.hpp"
#include "parser/type_checker.hpp"
#include <cstring>
#include <iostream>
#include <sstream>

#include <llvm/Support/raw_ostream.h>

static void printUsage(const char* progName) {
  std::cerr << "Usage: " << progName << " [options] [file]\n";
  std::cerr << "Options:\n";
  std::cerr << "  --dump-ast   Dump AST and exit (no code generation)\n";
  std::cerr << "  --emit-mlir  Emit Polang dialect MLIR instead of LLVM IR\n";
  std::cerr << "  --help       Show this help message\n";
}

int main(int argc, char** argv) {
  bool dumpAst = false;
  bool emitMlir = false;
  const char* inputFile = nullptr;

  // Parse command-line arguments
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--dump-ast") == 0) {
      dumpAst = true;
    } else if (std::strcmp(argv[i], "--emit-mlir") == 0) {
      emitMlir = true;
    } else if (std::strcmp(argv[i], "--help") == 0 ||
               std::strcmp(argv[i], "-h") == 0) {
      printUsage(argv[0]);
      return 0;
    } else if (argv[i][0] == '-') {
      std::cerr << "Unknown option: " << argv[i] << "\n";
      printUsage(argv[0]);
      return 1;
    } else {
      inputFile = argv[i];
    }
  }

  NBlock* ast = nullptr;

  if (inputFile != nullptr) {
    // File input mode
    ast = polang_parse_file(inputFile);
  } else {
    // Stdin mode
    std::stringstream buffer;
    buffer << std::cin.rdbuf();
    const std::string source = buffer.str();
    ast = polang_parse(source);
  }

  if (ast == nullptr) {
    return 1;
  }

  // Dump AST and exit if requested
  if (dumpAst) {
    ASTPrinter printer(std::cout);
    printer.print(*ast);
    return 0;
  }

  // Type checking
  const auto type_errors = polang_check_types(*ast);
  if (!type_errors.empty()) {
    return 1;
  }

  // MLIR backend
  polang::MLIRCodeGenContext context;

  if (!context.generateCode(*ast)) {
    std::cerr << "MLIR generation failed: " << context.getError() << "\n";
    return 1;
  }

  if (emitMlir) {
    // Just print the Polang dialect MLIR
    context.printMLIR(llvm::outs());
    return 0;
  }

  // Lower to standard dialects
  if (!context.lowerToStandard()) {
    std::cerr << "Lowering to standard failed: " << context.getError() << "\n";
    return 1;
  }

  // Lower to LLVM dialect
  if (!context.lowerToLLVM()) {
    std::cerr << "Lowering to LLVM failed: " << context.getError() << "\n";
    return 1;
  }

  // Print LLVM IR
  if (!context.printLLVMIR(llvm::outs())) {
    std::cerr << "LLVM IR generation failed: " << context.getError() << "\n";
    return 1;
  }

  return 0;
}
