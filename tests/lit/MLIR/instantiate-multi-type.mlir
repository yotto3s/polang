// RUN: %polang_opt %s | %FileCheck %s

// Test instantiate with multiple type bindings and mixed concrete types.

// CHECK:      polang.generic_func @pair<a, b>(%arg0: !polang.type_param<"a">, %arg1: !polang.type_param<"b">) -> !polang.type_param<"a"> {
// CHECK-NEXT:   polang.return %arg0 : !polang.type_param<"a">
// CHECK-NEXT: }

// CHECK:      polang.func @__polang_entry() -> !polang.integer<64, signed> {
// CHECK-NEXT:   %0 = polang.constant.integer 42 : !polang.integer<64, signed>
// CHECK-NEXT:   %1 = polang.constant.bool true : !polang.bool
// CHECK-NEXT:   %2 = polang.instantiate @pair<a = !polang.integer<64, signed>, b = !polang.bool>(%0, %1) : (!polang.integer<64, signed>, !polang.bool) -> !polang.integer<64, signed>
// CHECK-NEXT:   polang.return %2 : !polang.integer<64, signed>
// CHECK-NEXT: }
module {
  polang.generic_func @pair<a, b>(%x: !polang.type_param<"a">, %y: !polang.type_param<"b">) -> !polang.type_param<"a"> {
    polang.return %x : !polang.type_param<"a">
  }

  polang.func @__polang_entry() -> !polang.integer<64, signed> {
    %0 = polang.constant.integer 42 : !polang.integer<64, signed>
    %1 = polang.constant.bool true : !polang.bool
    %2 = polang.instantiate @pair<a = !polang.integer<64, signed>, b = !polang.bool>(%0, %1) : (!polang.integer<64, signed>, !polang.bool) -> !polang.integer<64, signed>
    polang.return %2 : !polang.integer<64, signed>
  }
}
