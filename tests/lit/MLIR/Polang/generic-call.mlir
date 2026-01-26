// RUN: %polang_opt %s | %polang_opt | %FileCheck %s

// Test CallOp with type arguments for generic functions

// =============================================================================
// Generic functions to call
// =============================================================================

polang.generic_func @identity<T>(%arg0: !polang.param<"T">) -> !polang.param<"T"> {
  polang.return %arg0 : !polang.param<"T">
}

polang.generic_func @add<T: numeric>(%arg0: !polang.param<"T", numeric>, %arg1: !polang.param<"T", numeric>) -> !polang.param<"T", numeric> {
  %r = polang.add %arg0, %arg1 : !polang.param<"T", numeric>, !polang.param<"T", numeric> -> !polang.param<"T", numeric>
  polang.return %r : !polang.param<"T", numeric>
}

polang.generic_func @first<A, B>(%arg0: !polang.param<"A">, %arg1: !polang.param<"B">) -> !polang.param<"A"> {
  polang.return %arg0 : !polang.param<"A">
}

// =============================================================================
// Calls with single type argument
// =============================================================================

polang.func @test_single_type_arg() -> !polang.integer<64, signed> {
  %c = polang.constant.integer 42 : !polang.integer<64, signed>
  // CHECK: {{.*}}polang.call @identity<[!polang.integer<64, signed>]>(%{{.*}}) : (!polang.integer<64, signed>) -> !polang.integer<64, signed>
  %r = polang.call @identity<[!polang.integer<64, signed>]>(%c) : (!polang.integer<64, signed>) -> !polang.integer<64, signed>
  polang.return %r : !polang.integer<64, signed>
}

polang.func @test_float_type_arg() -> !polang.float<64> {
  %c = polang.constant.float 3.14 : !polang.float<64>
  // CHECK: {{.*}}polang.call @identity<[!polang.float<64>]>(%{{.*}}) : (!polang.float<64>) -> !polang.float<64>
  %r = polang.call @identity<[!polang.float<64>]>(%c) : (!polang.float<64>) -> !polang.float<64>
  polang.return %r : !polang.float<64>
}

// =============================================================================
// Calls with numeric constraint
// =============================================================================

polang.func @test_add_integer() -> !polang.integer<64, signed> {
  %a = polang.constant.integer 10 : !polang.integer<64, signed>
  %b = polang.constant.integer 20 : !polang.integer<64, signed>
  // CHECK: {{.*}}polang.call @add<[!polang.integer<64, signed>]>(%{{.*}}, %{{.*}}) : (!polang.integer<64, signed>, !polang.integer<64, signed>) -> !polang.integer<64, signed>
  %r = polang.call @add<[!polang.integer<64, signed>]>(%a, %b) : (!polang.integer<64, signed>, !polang.integer<64, signed>) -> !polang.integer<64, signed>
  polang.return %r : !polang.integer<64, signed>
}

polang.func @test_add_float() -> !polang.float<64> {
  %a = polang.constant.float 1.5 : !polang.float<64>
  %b = polang.constant.float 2.5 : !polang.float<64>
  // CHECK: {{.*}}polang.call @add<[!polang.float<64>]>(%{{.*}}, %{{.*}}) : (!polang.float<64>, !polang.float<64>) -> !polang.float<64>
  %r = polang.call @add<[!polang.float<64>]>(%a, %b) : (!polang.float<64>, !polang.float<64>) -> !polang.float<64>
  polang.return %r : !polang.float<64>
}

// =============================================================================
// Calls with multiple type arguments
// =============================================================================

polang.func @test_multiple_type_args() -> !polang.integer<64, signed> {
  %a = polang.constant.integer 100 : !polang.integer<64, signed>
  %b = polang.constant.float 3.14 : !polang.float<64>
  // CHECK: {{.*}}polang.call @first<[!polang.integer<64, signed>, !polang.float<64>]>(%{{.*}}, %{{.*}}) : (!polang.integer<64, signed>, !polang.float<64>) -> !polang.integer<64, signed>
  %r = polang.call @first<[!polang.integer<64, signed>, !polang.float<64>]>(%a, %b) : (!polang.integer<64, signed>, !polang.float<64>) -> !polang.integer<64, signed>
  polang.return %r : !polang.integer<64, signed>
}

// =============================================================================
// Regular calls (without type arguments) still work
// =============================================================================

polang.func @helper(%arg0: !polang.integer<64, signed>) -> !polang.integer<64, signed> {
  polang.return %arg0 : !polang.integer<64, signed>
}

polang.func @test_regular_call() -> !polang.integer<64, signed> {
  %c = polang.constant.integer 42 : !polang.integer<64, signed>
  // CHECK: {{.*}}polang.call @helper(%{{.*}}) : (!polang.integer<64, signed>) -> !polang.integer<64, signed>
  %r = polang.call @helper(%c) : (!polang.integer<64, signed>) -> !polang.integer<64, signed>
  polang.return %r : !polang.integer<64, signed>
}
