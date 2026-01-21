#ifndef POLANG_PARSER_TEST_HELPER_HPP
#define POLANG_PARSER_TEST_HELPER_HPP

// Include gtest first to avoid conflicts with LLVM headers
#include <gtest/gtest.h>

// Standard library
#include <memory>
#include <string>

// clang-format off
// Parser headers (includes LLVM via node.hpp)
#include "parser/node.hpp"
#include "parser.hpp" // Must be after node.hpp - provides token constants
#include "parser/parser_api.hpp"
// clang-format on

// Token type shorthand
using token = yy::parser::token;

// Parse and return root block, fails test if null
inline std::unique_ptr<NBlock> parseOrFail(const std::string& source) {
  auto block = polang_parse(source);
  EXPECT_NE(block, nullptr) << "Failed to parse: " << source;
  return block;
}

// Get first statement as specific type
template <typename T> T* getFirstStatement(NBlock* block) {
  EXPECT_FALSE(block->statements.empty());
  if (block->statements.empty())
    return nullptr;
  T* stmt = dynamic_cast<T*>(block->statements[0].get());
  EXPECT_NE(stmt, nullptr) << "First statement is not of expected type";
  return stmt;
}

#endif // POLANG_PARSER_TEST_HELPER_HPP
