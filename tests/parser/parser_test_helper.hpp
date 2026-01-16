#ifndef POLANG_PARSER_TEST_HELPER_HPP
#define POLANG_PARSER_TEST_HELPER_HPP

// Include gtest first to avoid conflicts with LLVM headers
#include <gtest/gtest.h>

// Standard library
#include <string>

// clang-format off
// Parser headers (includes LLVM via node.hpp)
#include "parser/node.hpp"
#include "parser.hpp" // Must be after node.hpp - provides token constants
#include "parser/parser_api.hpp"
// clang-format on

// Parse and return root block, fails test if null
inline NBlock* parseOrFail(const std::string& source) {
  NBlock* block = polang_parse(source);
  EXPECT_NE(block, nullptr) << "Failed to parse: " << source;
  return block;
}

// Get first statement as specific type
template <typename T> T* getFirstStatement(NBlock* block) {
  EXPECT_FALSE(block->statements.empty());
  if (block->statements.empty())
    return nullptr;
  T* stmt = dynamic_cast<T*>(block->statements[0]);
  EXPECT_NE(stmt, nullptr) << "First statement is not of expected type";
  return stmt;
}

#endif // POLANG_PARSER_TEST_HELPER_HPP
