// RUN: %polang_opt %s | %FileCheck %s

// Test round-trip of a simple generic function (identity) with no bounds.

// CHECK:      polang.generic_func @identity<a>(%arg0: !polang.type_param<"a">) -> !polang.type_param<"a"> {
// CHECK-NEXT:   polang.return %arg0 : !polang.type_param<"a">
// CHECK-NEXT: }
module {
  polang.generic_func @identity<a>(%x: !polang.type_param<"a">) -> !polang.type_param<"a"> {
    polang.return %x : !polang.type_param<"a">
  }
}
