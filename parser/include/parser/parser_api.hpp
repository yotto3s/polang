#ifndef POLANG_PARSER_API_HPP
#define POLANG_PARSER_API_HPP

#include <string>

class NBlock;

NBlock* polang_parse(const std::string& source);
NBlock* polang_parse_file(const char* filename);

#endif // POLANG_PARSER_API_HPP
