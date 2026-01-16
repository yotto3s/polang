#include "parser/parser_api.hpp"
#include "parser/node.hpp"
#include "parser/type_checker.hpp"

#include <fstream>
#include <iostream>
#include <sstream>

// Flex buffer type
typedef struct yy_buffer_state* YY_BUFFER_STATE;

// Flex functions for string scanning
extern YY_BUFFER_STATE yy_scan_string(const char* str);
extern void yy_delete_buffer(YY_BUFFER_STATE buffer);
extern void polang_reset_lexer_location();

// Bison parser function and result
extern int yyparse();
extern NBlock* programBlock;

NBlock* polang_parse(const std::string& source) {
  polang_reset_lexer_location();
  programBlock = nullptr; // Reset before parsing
  const YY_BUFFER_STATE buffer = yy_scan_string(source.c_str());
  const int parseResult = yyparse();
  yy_delete_buffer(buffer);
  // Return nullptr if parsing failed
  if (parseResult != 0) {
    return nullptr;
  }
  return programBlock;
}

NBlock* polang_parse_file(const char* filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cerr << "ERROR: Cannot open file: " << filename << std::endl;
    return nullptr;
  }
  std::stringstream buffer;
  buffer << file.rdbuf();
  return polang_parse(buffer.str());
}

std::vector<TypeCheckError> polang_check_types(const NBlock& ast) {
  TypeChecker checker;
  return checker.check(ast);
}
