//===- runtime.c - Polang runtime library implementation --------*- C -*-===//
//
// Implementation of Polang runtime library functions.
//
//===----------------------------------------------------------------------===//

#include "runtime.h"
#include <stdio.h>
#include <stdlib.h>

void __polang_runtime_error(const char* message, int line, int column) {
  if (line > 0 && column > 0) {
    fprintf(stderr, "Runtime error: %s at line %d, column %d\n", message, line,
            column);
  } else {
    fprintf(stderr, "Runtime error: %s\n", message);
  }
  exit(1);
}
