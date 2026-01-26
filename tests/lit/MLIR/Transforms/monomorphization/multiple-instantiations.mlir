// RUN: %polang_opt %s -polang-monomorphize | %FileCheck %s

// Test monomorphization creates multiple specialized functions for different type args

// CHECK: module {

// Original generic function should remain
// CHECK-NEXT:   polang.generic_func @identity<T>(%arg0: !polang.param<"T">) -> !polang.param<"T"> {
// CHECK-NEXT:     polang.return %arg0 : !polang.param<"T">
// CHECK-NEXT:   }
polang.generic_func @identity<T>(%arg0: !polang.param<"T">) -> !polang.param<"T"> {
  polang.return %arg0 : !polang.param<"T">
}

// Two specialized function markers should be created
// CHECK-NEXT:   polang.specialized_func @identity$i64 from @identity<[!polang.integer<64, signed>]>
// CHECK-NEXT:   polang.specialized_func @identity$f64 from @identity<[!polang.float<64>]>

// Caller with i64
// CHECK-NEXT:   polang.func @call_with_int() -> !polang.integer<64, signed> {
// CHECK-NEXT:     %0 = polang.constant.integer 42 : !polang.integer<64, signed>
// CHECK-NEXT:     %1 = polang.call @identity$i64(%0) : (!polang.integer<64, signed>) -> !polang.integer<64, signed>
// CHECK-NEXT:     polang.return %1 : !polang.integer<64, signed>
// CHECK-NEXT:   }
polang.func @call_with_int() -> !polang.integer<64, signed> {
  %c42 = polang.constant.integer 42 : !polang.integer<64, signed>
  %result = polang.call @identity<[!polang.integer<64, signed>]>(%c42) : (!polang.integer<64, signed>) -> !polang.integer<64, signed>
  polang.return %result : !polang.integer<64, signed>
}

// Caller with f64
// CHECK-NEXT:   polang.func @call_with_float() -> !polang.float<64> {
// CHECK-NEXT:     %0 = polang.constant.float 3.140000e+00 : !polang.float<64>
// CHECK-NEXT:     %1 = polang.call @identity$f64(%0) : (!polang.float<64>) -> !polang.float<64>
// CHECK-NEXT:     polang.return %1 : !polang.float<64>
// CHECK-NEXT:   }
polang.func @call_with_float() -> !polang.float<64> {
  %c3_14 = polang.constant.float 3.14 : !polang.float<64>
  %result = polang.call @identity<[!polang.float<64>]>(%c3_14) : (!polang.float<64>) -> !polang.float<64>
  polang.return %result : !polang.float<64>
}

// CHECK-NEXT: }
