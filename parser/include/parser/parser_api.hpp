#ifndef POLANG_PARSER_API_HPP
#define POLANG_PARSER_API_HPP

#include <memory>
#include <string>
#include <vector>

class NBlock;
struct TypeCheckError;

std::unique_ptr<NBlock> polang_parse(const std::string& source);
std::unique_ptr<NBlock> polang_parse_file(const char* filename);

// Type checking - returns list of errors (empty if no errors)
std::vector<TypeCheckError> polang_check_types(const NBlock& ast);

#endif // POLANG_PARSER_API_HPP
