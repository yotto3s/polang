// RUN: %polang_opt %s | %FileCheck %s

// Test round-trip of a generic function with multiple type parameters.

// CHECK:      polang.generic_func @pair<a, b>(%arg0: !polang.type_param<"a">, %arg1: !polang.type_param<"b">) -> !polang.type_param<"a"> {
// CHECK-NEXT:   polang.return %arg0 : !polang.type_param<"a">
// CHECK-NEXT: }
module {
  polang.generic_func @pair<a, b>(%x: !polang.type_param<"a">, %y: !polang.type_param<"b">) -> !polang.type_param<"a"> {
    polang.return %x : !polang.type_param<"a">
  }
}
