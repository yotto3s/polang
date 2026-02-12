//===- runtime_helpers.c - Runtime helper functions for Polang ---*- C -*-===//
//
// Runtime helper functions callable from JIT-compiled code.
//
//===----------------------------------------------------------------------===//

#include <stdio.h>
#include <stdlib.h>

/// Runtime error handler for integer division by zero.
/// Prints an error message with source location and exits.
/// @param line Source line number
/// @param column Source column number
void __polang_divzero_error(int line, int column) {
  fprintf(stderr, "Runtime error: integer division by zero at line %d, column %d\n",
          line, column);
  exit(1);
}
