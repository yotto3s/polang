#ifndef POLANG_RUNTIME_H
#define POLANG_RUNTIME_H

#ifdef __cplusplus
extern "C" {
#endif

void __polang_runtime_error(const char* message, int line, int column);

#ifdef __cplusplus
}
#endif

#endif // POLANG_RUNTIME_H
