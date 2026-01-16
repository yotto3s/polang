#ifndef POLANG_PARSER_API_HPP
#define POLANG_PARSER_API_HPP

#include <string>

class NBlock;

NBlock *polang_parse(const std::string &source);

#endif // POLANG_PARSER_API_HPP
