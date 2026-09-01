// RUN: %polang_opt %s | %FileCheck %s

// Test instantiate with multiple type bindings and mixed concrete types.

// CHECK:      polang.generic_func @pair<a, b>(%arg0: !polang.type_param<"a">, %arg1: !polang.type_param<"b">) -> !polang.type_param<"a"> {
// CHECK-NEXT:   polang.return %arg0 : !polang.type_param<"a">
// CHECK-NEXT: }

// CHECK:      polang.func @__polang_entry() -> si64 {
// CHECK-NEXT:   %0 = polang.constant.integer 42 : si64
// CHECK-NEXT:   %true = arith.constant true
// CHECK-NEXT:   %1 = polang.instantiate @pair<a = si64, b = i1>(%0, %true) : (si64, i1) -> si64
// CHECK-NEXT:   polang.return %1 : si64
// CHECK-NEXT: }
module {
  polang.generic_func @pair<a, b>(%x: !polang.type_param<"a">, %y: !polang.type_param<"b">) -> !polang.type_param<"a"> {
    polang.return %x : !polang.type_param<"a">
  }

  polang.func @__polang_entry() -> si64 {
    %0 = polang.constant.integer 42 : si64
    %1 = arith.constant true
    %2 = polang.instantiate @pair<a = si64, b = i1>(%0, %1) : (si64, i1) -> si64
    polang.return %2 : si64
  }
}
