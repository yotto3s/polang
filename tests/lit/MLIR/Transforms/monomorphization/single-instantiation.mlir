// RUN: %polang_opt %s -polang-monomorphize | %FileCheck %s

// Test monomorphization creates a specialized function for a single instantiation

// CHECK: module {

// Original generic function should remain
// CHECK-NEXT:   polang.generic_func @identity<T>(%arg0: !polang.param<"T">) -> !polang.param<"T"> {
// CHECK-NEXT:     polang.return %arg0 : !polang.param<"T">
// CHECK-NEXT:   }
polang.generic_func @identity<T>(%arg0: !polang.param<"T">) -> !polang.param<"T"> {
  polang.return %arg0 : !polang.param<"T">
}

// Specialized function marker should be created
// CHECK-NEXT:   polang.specialized_func @identity$i64 from @identity<[!polang.integer<64, signed>]>

// Caller function should have its call site updated
// CHECK-NEXT:   polang.func @caller() -> !polang.integer<64, signed> {
// CHECK-NEXT:     %0 = polang.constant.integer 42 : !polang.integer<64, signed>
// Call should use specialized name without type_args
// CHECK-NEXT:     %1 = polang.call @identity$i64(%0) : (!polang.integer<64, signed>) -> !polang.integer<64, signed>
// CHECK-NEXT:     polang.return %1 : !polang.integer<64, signed>
// CHECK-NEXT:   }
polang.func @caller() -> !polang.integer<64, signed> {
  %c42 = polang.constant.integer 42 : !polang.integer<64, signed>
  %result = polang.call @identity<[!polang.integer<64, signed>]>(%c42) : (!polang.integer<64, signed>) -> !polang.integer<64, signed>
  polang.return %result : !polang.integer<64, signed>
}

// CHECK-NEXT: }
