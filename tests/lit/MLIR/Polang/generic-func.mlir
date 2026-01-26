// RUN: %polang_opt %s | %polang_opt | %FileCheck %s

// Test GenericFuncOp parsing and printing

// =============================================================================
// Single type parameter with various constraints
// =============================================================================

// CHECK: polang.generic_func @identity<T>(%arg0: !polang.param<"T">) -> !polang.param<"T">
polang.generic_func @identity<T>(%arg0: !polang.param<"T">) -> !polang.param<"T"> {
  polang.return %arg0 : !polang.param<"T">
}

// CHECK: polang.generic_func @identity_numeric<N: numeric>(%arg0: !polang.param<"N", numeric>) -> !polang.param<"N", numeric>
polang.generic_func @identity_numeric<N: numeric>(%arg0: !polang.param<"N", numeric>) -> !polang.param<"N", numeric> {
  polang.return %arg0 : !polang.param<"N", numeric>
}

// CHECK: polang.generic_func @identity_integer<I: integer>(%arg0: !polang.param<"I", integer>) -> !polang.param<"I", integer>
polang.generic_func @identity_integer<I: integer>(%arg0: !polang.param<"I", integer>) -> !polang.param<"I", integer> {
  polang.return %arg0 : !polang.param<"I", integer>
}

// CHECK: polang.generic_func @identity_float<F: float>(%arg0: !polang.param<"F", float>) -> !polang.param<"F", float>
polang.generic_func @identity_float<F: float>(%arg0: !polang.param<"F", float>) -> !polang.param<"F", float> {
  polang.return %arg0 : !polang.param<"F", float>
}

// =============================================================================
// Multiple type parameters
// =============================================================================

// CHECK: polang.generic_func @pair<A, B>(%arg0: !polang.param<"A">, %arg1: !polang.param<"B">) -> !polang.param<"A">
polang.generic_func @pair<A, B>(%arg0: !polang.param<"A">, %arg1: !polang.param<"B">) -> !polang.param<"A"> {
  polang.return %arg0 : !polang.param<"A">
}

// CHECK: polang.generic_func @add<T: numeric>(%arg0: !polang.param<"T", numeric>, %arg1: !polang.param<"T", numeric>) -> !polang.param<"T", numeric>
polang.generic_func @add<T: numeric>(%arg0: !polang.param<"T", numeric>, %arg1: !polang.param<"T", numeric>) -> !polang.param<"T", numeric> {
  %r = polang.add %arg0, %arg1 : !polang.param<"T", numeric>, !polang.param<"T", numeric> -> !polang.param<"T", numeric>
  polang.return %r : !polang.param<"T", numeric>
}

// =============================================================================
// Mixed constraints
// =============================================================================

// CHECK: polang.generic_func @mixed<A, N: numeric, I: integer>(%arg0: !polang.param<"A">, %arg1: !polang.param<"N", numeric>, %arg2: !polang.param<"I", integer>) -> !polang.param<"A">
polang.generic_func @mixed<A, N: numeric, I: integer>(%arg0: !polang.param<"A">, %arg1: !polang.param<"N", numeric>, %arg2: !polang.param<"I", integer>) -> !polang.param<"A"> {
  polang.return %arg0 : !polang.param<"A">
}
