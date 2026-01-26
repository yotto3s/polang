// RUN: %polang_opt %s | %polang_opt | %FileCheck %s

// Test SpecializedFuncOp parsing and printing

// CHECK: module {

// =============================================================================
// Generic function to specialize from
// =============================================================================

polang.generic_func @identity<T>(%arg0: !polang.param<"T">) -> !polang.param<"T"> {
  polang.return %arg0 : !polang.param<"T">
}

polang.generic_func @add<T: numeric>(%arg0: !polang.param<"T", numeric>, %arg1: !polang.param<"T", numeric>) -> !polang.param<"T", numeric> {
  %r = polang.add %arg0, %arg1 : !polang.param<"T", numeric>, !polang.param<"T", numeric> -> !polang.param<"T", numeric>
  polang.return %r : !polang.param<"T", numeric>
}

polang.generic_func @pair<A, B>(%arg0: !polang.param<"A">, %arg1: !polang.param<"B">) -> !polang.param<"A"> {
  polang.return %arg0 : !polang.param<"A">
}

// =============================================================================
// Specializations with single type argument
// =============================================================================

// CHECK: {{.*}}polang.specialized_func @identity$i64 from @identity<[!polang.integer<64, signed>]>
polang.specialized_func @identity$i64 from @identity<[!polang.integer<64, signed>]>

// CHECK: {{.*}}polang.specialized_func @identity$f64 from @identity<[!polang.float<64>]>
polang.specialized_func @identity$f64 from @identity<[!polang.float<64>]>

// CHECK: {{.*}}polang.specialized_func @identity$bool from @identity<[!polang.bool]>
polang.specialized_func @identity$bool from @identity<[!polang.bool]>

// =============================================================================
// Specializations with numeric constraint
// =============================================================================

// CHECK: {{.*}}polang.specialized_func @add$i64 from @add<[!polang.integer<64, signed>]>
polang.specialized_func @add$i64 from @add<[!polang.integer<64, signed>]>

// CHECK: {{.*}}polang.specialized_func @add$f64 from @add<[!polang.float<64>]>
polang.specialized_func @add$f64 from @add<[!polang.float<64>]>

// =============================================================================
// Specializations with multiple type arguments
// =============================================================================

// CHECK: {{.*}}polang.specialized_func @pair$i64_f64 from @pair<[!polang.integer<64, signed>, !polang.float<64>]>
polang.specialized_func @pair$i64_f64 from @pair<[!polang.integer<64, signed>, !polang.float<64>]>

// CHECK: {{.*}}polang.specialized_func @pair$bool_i32 from @pair<[!polang.bool, !polang.integer<32, signed>]>
polang.specialized_func @pair$bool_i32 from @pair<[!polang.bool, !polang.integer<32, signed>]>

// CHECK: }
