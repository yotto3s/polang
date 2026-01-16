#include "compiler/codegen.hpp"
#include "parser/node.hpp"
#include "parser/parser_api.hpp"
#include <iostream>
#include <sstream>

int main(int argc, char** argv) {
  std::stringstream buffer;
  buffer << std::cin.rdbuf();
  std::string source = buffer.str();

  NBlock* ast = polang_parse(source);
  if (!ast) {
    return 1;
  }

  CodeGenContext context;
  context.generateCode(*ast);
  context.printIR(llvm::outs());

  return 0;
}
