//===- runtime.h - Polang runtime library header ----------------*- C -*-===//
//
// Public interface for the Polang runtime library.
//
//===----------------------------------------------------------------------===//

#ifndef POLANG_RUNTIME_H
#define POLANG_RUNTIME_H

#ifdef __cplusplus
extern "C" {
#endif

/// Runtime error handler - prints error message with location and exits
void __polang_runtime_error(const char* message, int line, int column);

#ifdef __cplusplus
}
#endif

#endif // POLANG_RUNTIME_H
