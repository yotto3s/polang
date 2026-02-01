#ifndef POLANG_JIT_RESULT_HPP
#define POLANG_JIT_RESULT_HPP

#include <cstdint>
#include <variant>

namespace polang {

/// Type-safe variant for JIT execution results.
/// std::monostate represents void / no value.
using JitResult =
    std::variant<std::monostate, int64_t, double, float, bool, int8_t, int16_t>;

} // namespace polang

#endif // POLANG_JIT_RESULT_HPP
