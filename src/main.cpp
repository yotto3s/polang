#include "codegen.hpp"
#include "node.hpp"

extern NBlock *programBlock;
extern int yyparse();

int main(int argc, char **argv) {
  yyparse();
  CodeGenContext context;
  context.generateCode(*programBlock);
  context.runCode();
  return 0;
}
