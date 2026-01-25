// RUN: %polang_opt %s | %polang_opt | %FileCheck %s

// Test polang_ast function operations

// CHECK: module {

// Test function declaration with concrete types
// CHECK-NEXT:   polang_ast.func @add(%arg0: !polang.integer<64, signed>, %arg1: !polang.integer<64, signed>) -> !polang.integer<64, signed> {
// CHECK-NEXT:     %0 = polang_ast.add %arg0, %arg1 : !polang.integer<64, signed>, !polang.integer<64, signed> -> !polang.integer<64, signed>
// CHECK-NEXT:     polang_ast.return %0 : !polang.integer<64, signed>
// CHECK-NEXT:   }
polang_ast.func @add(%a: !polang.integer<64, signed>, %b: !polang.integer<64, signed>) -> !polang.integer<64, signed> {
  %0 = polang_ast.add %a, %b : !polang.integer<64, signed>, !polang.integer<64, signed> -> !polang.integer<64, signed>
  polang_ast.return %0 : !polang.integer<64, signed>
}

// Test function with type variables
// CHECK-NEXT:   polang_ast.func @generic_add(%arg0: !polang_ast.typevar<1, integer>, %arg1: !polang_ast.typevar<1, integer>) -> !polang_ast.typevar<1, integer> {
// CHECK-NEXT:     %0 = polang_ast.add %arg0, %arg1 : !polang_ast.typevar<1, integer>, !polang_ast.typevar<1, integer> -> !polang_ast.typevar<1, integer>
// CHECK-NEXT:     polang_ast.return %0 : !polang_ast.typevar<1, integer>
// CHECK-NEXT:   }
polang_ast.func @generic_add(%a: !polang_ast.typevar<1, integer>, %b: !polang_ast.typevar<1, integer>) -> !polang_ast.typevar<1, integer> {
  %0 = polang_ast.add %a, %b : !polang_ast.typevar<1, integer>, !polang_ast.typevar<1, integer> -> !polang_ast.typevar<1, integer>
  polang_ast.return %0 : !polang_ast.typevar<1, integer>
}

// Test function call
// CHECK-NEXT:   polang_ast.func @main() -> !polang.integer<64, signed> {
// CHECK-NEXT:     %0 = polang_ast.constant.integer 10 : !polang.integer<64, signed>
// CHECK-NEXT:     %1 = polang_ast.constant.integer 20 : !polang.integer<64, signed>
// CHECK-NEXT:     %2 = polang_ast.call @add(%0, %1) : (!polang.integer<64, signed>, !polang.integer<64, signed>) -> !polang.integer<64, signed>
// CHECK-NEXT:     polang_ast.return %2 : !polang.integer<64, signed>
// CHECK-NEXT:   }
polang_ast.func @main() -> !polang.integer<64, signed> {
  %0 = polang_ast.constant.integer 10 : !polang.integer<64, signed>
  %1 = polang_ast.constant.integer 20 : !polang.integer<64, signed>
  %2 = polang_ast.call @add(%0, %1) : (!polang.integer<64, signed>, !polang.integer<64, signed>) -> !polang.integer<64, signed>
  polang_ast.return %2 : !polang.integer<64, signed>
}

// CHECK-NEXT: }
