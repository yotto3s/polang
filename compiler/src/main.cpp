#include "compiler/codegen.hpp"
#include "parser/node.hpp"
#include "parser/parser_api.hpp"
#include <iostream>
#include <sstream>

int main(int argc, char** argv) {
  NBlock* ast = nullptr;

  if (argc > 1) {
    // File input mode
    ast = polang_parse_file(argv[1]);
  } else {
    // Stdin mode
    std::stringstream buffer;
    buffer << std::cin.rdbuf();
    const std::string source = buffer.str();
    ast = polang_parse(source);
  }

  if (!ast) {
    return 1;
  }

  CodeGenContext context;
  context.generateCode(*ast);
  context.printIR(llvm::outs());

  return 0;
}
