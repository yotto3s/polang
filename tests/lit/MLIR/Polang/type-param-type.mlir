// RUN: %polang_opt %s | %polang_opt | %FileCheck %s

// Test TypeParamType parsing and printing

// =============================================================================
// TypeParamType with explicit constraints
// =============================================================================

// CHECK: polang.func @type_param_any(%arg0: !polang.param<"T">) -> !polang.param<"T">
polang.func @type_param_any(%arg0: !polang.param<"T">) -> !polang.param<"T"> {
  polang.return %arg0 : !polang.param<"T">
}

// CHECK: polang.func @type_param_numeric(%arg0: !polang.param<"N", numeric>) -> !polang.param<"N", numeric>
polang.func @type_param_numeric(%arg0: !polang.param<"N", numeric>) -> !polang.param<"N", numeric> {
  polang.return %arg0 : !polang.param<"N", numeric>
}

// CHECK: polang.func @type_param_integer(%arg0: !polang.param<"I", integer>) -> !polang.param<"I", integer>
polang.func @type_param_integer(%arg0: !polang.param<"I", integer>) -> !polang.param<"I", integer> {
  polang.return %arg0 : !polang.param<"I", integer>
}

// CHECK: polang.func @type_param_float(%arg0: !polang.param<"F", float>) -> !polang.param<"F", float>
polang.func @type_param_float(%arg0: !polang.param<"F", float>) -> !polang.param<"F", float> {
  polang.return %arg0 : !polang.param<"F", float>
}

// =============================================================================
// TypeParamType in arithmetic operations
// =============================================================================

// CHECK: polang.func @add_with_param
polang.func @add_with_param(%a: !polang.param<"T", numeric>, %b: !polang.param<"T", numeric>) -> !polang.param<"T", numeric> {
  // CHECK: polang.add %arg0, %arg1 : !polang.param<"T", numeric>, !polang.param<"T", numeric> -> !polang.param<"T", numeric>
  %r = polang.add %a, %b : !polang.param<"T", numeric>, !polang.param<"T", numeric> -> !polang.param<"T", numeric>
  polang.return %r : !polang.param<"T", numeric>
}

// =============================================================================
// Multiple type parameters with different names
// =============================================================================

// CHECK: polang.func @two_params(%arg0: !polang.param<"A">, %arg1: !polang.param<"B", integer>) -> !polang.param<"A">
polang.func @two_params(%a: !polang.param<"A">, %b: !polang.param<"B", integer>) -> !polang.param<"A"> {
  polang.return %a : !polang.param<"A">
}
