// RUN: %polang_opt %s | %FileCheck %s

// Test round-trip of a generic function with trait bounds and arithmetic ops.

// CHECK:      polang.generic_func @add<a: Numeric>(%arg0: !polang.type_param<"a">, %arg1: !polang.type_param<"a">) -> !polang.type_param<"a"> {
// CHECK-NEXT:   %0 = polang.add %arg0, %arg1 : !polang.type_param<"a">, !polang.type_param<"a"> -> !polang.type_param<"a">
// CHECK-NEXT:   polang.return %0 : !polang.type_param<"a">
// CHECK-NEXT: }
module {
  polang.generic_func @add<a: Numeric>(%x: !polang.type_param<"a">, %y: !polang.type_param<"a">) -> !polang.type_param<"a"> {
    %0 = polang.add %x, %y : !polang.type_param<"a">, !polang.type_param<"a"> -> !polang.type_param<"a">
    polang.return %0 : !polang.type_param<"a">
  }
}
