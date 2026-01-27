// RUN: %polang_opt %s -convert-polang-to-standard | %FileCheck %s

// Test lowering of SpecializedFuncOp to func.func with body from GenericFuncOp

// CHECK: module {

// GenericFuncOp should be erased after lowering
// CHECK-NOT: polang.generic_func
polang.generic_func @identity<T>(%arg0: !polang.param<"T">) -> !polang.param<"T"> {
  polang.return %arg0 : !polang.param<"T">
}

// SpecializedFuncOp should be lowered to func.func with concrete types
// CHECK-NOT: polang.specialized_func
// CHECK: func.func @identity$i64(%arg0: i64) -> i64 {
// CHECK-NEXT:   return %arg0 : i64
// CHECK-NEXT: }
polang.specialized_func @identity$i64 from @identity<[!polang.integer<64, signed>]>

// Caller function should have its call lowered
// CHECK: func.func @caller() -> i64 {
// CHECK:   %[[C42:.*]] = arith.constant 42 : i64
// CHECK:   %[[RESULT:.*]] = call @identity$i64(%[[C42]]) : (i64) -> i64
// CHECK:   return %[[RESULT]] : i64
// CHECK: }
polang.func @caller() -> !polang.integer<64, signed> {
  %c42 = polang.constant.integer 42 : !polang.integer<64, signed>
  %result = polang.call @identity$i64(%c42) : (!polang.integer<64, signed>) -> !polang.integer<64, signed>
  polang.return %result : !polang.integer<64, signed>
}

// CHECK: }
