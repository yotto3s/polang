// RUN: %polang_opt %s | %FileCheck %s

// Test round-trip of instantiate op calling generic functions.

// CHECK:      polang.generic_func @identity<a>(%arg0: !polang.type_param<"a">) -> !polang.type_param<"a"> {
// CHECK-NEXT:   polang.return %arg0 : !polang.type_param<"a">
// CHECK-NEXT: }

// CHECK:      polang.generic_func @add<a: Numeric>(%arg0: !polang.type_param<"a">, %arg1: !polang.type_param<"a">) -> !polang.type_param<"a"> {
// CHECK-NEXT:   %0 = polang.add %arg0, %arg1 : !polang.type_param<"a">, !polang.type_param<"a"> -> !polang.type_param<"a">
// CHECK-NEXT:   polang.return %0 : !polang.type_param<"a">
// CHECK-NEXT: }

// CHECK:      polang.func @__polang_entry() -> si64 {
// CHECK-NEXT:   %0 = polang.constant.integer 1 : si64
// CHECK-NEXT:   %1 = polang.constant.integer 2 : si64
// CHECK-NEXT:   %2 = polang.instantiate @add<a = si64>(%0, %1) : (si64, si64) -> si64
// CHECK-NEXT:   %3 = polang.instantiate @identity<a = si64>(%2) : (si64) -> si64
// CHECK-NEXT:   polang.return %3 : si64
// CHECK-NEXT: }
module {
  polang.generic_func @identity<a>(%x: !polang.type_param<"a">) -> !polang.type_param<"a"> {
    polang.return %x : !polang.type_param<"a">
  }

  polang.generic_func @add<a: Numeric>(%x: !polang.type_param<"a">, %y: !polang.type_param<"a">) -> !polang.type_param<"a"> {
    %0 = polang.add %x, %y : !polang.type_param<"a">, !polang.type_param<"a"> -> !polang.type_param<"a">
    polang.return %0 : !polang.type_param<"a">
  }

  polang.func @__polang_entry() -> si64 {
    %0 = polang.constant.integer 1 : si64
    %1 = polang.constant.integer 2 : si64
    %2 = polang.instantiate @add<a = si64>(%0, %1) : (si64, si64) -> si64
    %3 = polang.instantiate @identity<a = si64>(%2) : (si64) -> si64
    polang.return %3 : si64
  }
}
