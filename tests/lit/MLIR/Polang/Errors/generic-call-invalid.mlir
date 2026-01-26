// RUN: %polang_opt %s -split-input-file -verify-diagnostics

// =============================================================================
// Setup: Generic functions to call
// =============================================================================

polang.generic_func @identity<T>(%arg0: !polang.param<"T">) -> !polang.param<"T"> {
  polang.return %arg0 : !polang.param<"T">
}

polang.generic_func @add<T: numeric>(%arg0: !polang.param<"T", numeric>, %arg1: !polang.param<"T", numeric>) -> !polang.param<"T", numeric> {
  %r = polang.add %arg0, %arg1 : !polang.param<"T", numeric>, !polang.param<"T", numeric> -> !polang.param<"T", numeric>
  polang.return %r : !polang.param<"T", numeric>
}

polang.generic_func @int_only<T: integer>(%arg0: !polang.param<"T", integer>) -> !polang.param<"T", integer> {
  polang.return %arg0 : !polang.param<"T", integer>
}

// =============================================================================
// Test: Type argument violates numeric constraint
// =============================================================================

polang.func @call_add_with_bool() -> !polang.bool {
  %c = polang.constant.bool true : !polang.bool
  // expected-error @below {{type argument '!polang.bool' does not satisfy 'numeric' constraint for type parameter 'T'}}
  %r = polang.call @add<[!polang.bool]>(%c, %c) : (!polang.bool, !polang.bool) -> !polang.bool
  polang.return %r : !polang.bool
}

// -----

polang.generic_func @int_only<T: integer>(%arg0: !polang.param<"T", integer>) -> !polang.param<"T", integer> {
  polang.return %arg0 : !polang.param<"T", integer>
}

// =============================================================================
// Test: Type argument violates integer constraint
// =============================================================================

polang.func @call_int_only_with_float() -> !polang.float<64> {
  %c = polang.constant.float 3.14 : !polang.float<64>
  // expected-error @below {{type argument '!polang.float<64>' does not satisfy 'integer' constraint for type parameter 'T'}}
  %r = polang.call @int_only<[!polang.float<64>]>(%c) : (!polang.float<64>) -> !polang.float<64>
  polang.return %r : !polang.float<64>
}

// -----

polang.generic_func @pair<A, B>(%arg0: !polang.param<"A">, %arg1: !polang.param<"B">) -> !polang.param<"A"> {
  polang.return %arg0 : !polang.param<"A">
}

// =============================================================================
// Test: Wrong number of type arguments
// =============================================================================

polang.func @call_with_wrong_count() -> !polang.integer<64, signed> {
  %a = polang.constant.integer 42 : !polang.integer<64, signed>
  %b = polang.constant.float 3.14 : !polang.float<64>
  // expected-error @below {{expects 2 type argument(s) but got 1}}
  %r = polang.call @pair<[!polang.integer<64, signed>]>(%a, %b) : (!polang.integer<64, signed>, !polang.float<64>) -> !polang.integer<64, signed>
  polang.return %r : !polang.integer<64, signed>
}

// -----

polang.generic_func @identity<T>(%arg0: !polang.param<"T">) -> !polang.param<"T"> {
  polang.return %arg0 : !polang.param<"T">
}

// =============================================================================
// Test: Type arguments provided but callee is not a generic function
// =============================================================================

polang.func @regular_func(%arg0: !polang.integer<64, signed>) -> !polang.integer<64, signed> {
  polang.return %arg0 : !polang.integer<64, signed>
}

polang.func @call_regular_with_type_args() -> !polang.integer<64, signed> {
  %c = polang.constant.integer 42 : !polang.integer<64, signed>
  // expected-error @below {{type arguments provided but 'regular_func' is not a generic function}}
  %r = polang.call @regular_func<[!polang.integer<64, signed>]>(%c) : (!polang.integer<64, signed>) -> !polang.integer<64, signed>
  polang.return %r : !polang.integer<64, signed>
}
