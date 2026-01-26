// RUN: %polang_opt %s -polang-monomorphize | %FileCheck %s

// Test that multiple calls with the same type args only create one specialized function

// CHECK: module {

// CHECK-NEXT:   polang.generic_func @identity<T>(%arg0: !polang.param<"T">) -> !polang.param<"T"> {
// CHECK-NEXT:     polang.return %arg0 : !polang.param<"T">
// CHECK-NEXT:   }
polang.generic_func @identity<T>(%arg0: !polang.param<"T">) -> !polang.param<"T"> {
  polang.return %arg0 : !polang.param<"T">
}

// Only one specialized function for i64, not two
// CHECK-NEXT:   polang.specialized_func @identity$i64 from @identity<[!polang.integer<64, signed>]>
// CHECK-NOT:    polang.specialized_func @identity$i64

// First caller
// CHECK-NEXT:   polang.func @caller1() -> !polang.integer<64, signed> {
// CHECK-NEXT:     %0 = polang.constant.integer 1 : !polang.integer<64, signed>
// CHECK-NEXT:     %1 = polang.call @identity$i64(%0) : (!polang.integer<64, signed>) -> !polang.integer<64, signed>
// CHECK-NEXT:     polang.return %1 : !polang.integer<64, signed>
// CHECK-NEXT:   }
polang.func @caller1() -> !polang.integer<64, signed> {
  %c1 = polang.constant.integer 1 : !polang.integer<64, signed>
  %result = polang.call @identity<[!polang.integer<64, signed>]>(%c1) : (!polang.integer<64, signed>) -> !polang.integer<64, signed>
  polang.return %result : !polang.integer<64, signed>
}

// Second caller with same type args
// CHECK-NEXT:   polang.func @caller2() -> !polang.integer<64, signed> {
// CHECK-NEXT:     %0 = polang.constant.integer 2 : !polang.integer<64, signed>
// CHECK-NEXT:     %1 = polang.call @identity$i64(%0) : (!polang.integer<64, signed>) -> !polang.integer<64, signed>
// CHECK-NEXT:     polang.return %1 : !polang.integer<64, signed>
// CHECK-NEXT:   }
polang.func @caller2() -> !polang.integer<64, signed> {
  %c2 = polang.constant.integer 2 : !polang.integer<64, signed>
  %result = polang.call @identity<[!polang.integer<64, signed>]>(%c2) : (!polang.integer<64, signed>) -> !polang.integer<64, signed>
  polang.return %result : !polang.integer<64, signed>
}

// CHECK-NEXT: }
