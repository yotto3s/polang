// RUN: %polang_opt %s -polang-type-check | %FileCheck %s

// Test that valid binary operations pass type checking

// CHECK: module {

// CHECK-NEXT:   polang_ast.func @add_integers() -> !polang.integer<64, signed> {
// CHECK-NEXT:     %0 = polang_ast.constant.integer 1 : !polang.integer<64, signed>
// CHECK-NEXT:     %1 = polang_ast.constant.integer 2 : !polang.integer<64, signed>
// CHECK-NEXT:     %2 = polang_ast.add %0, %1 : !polang.integer<64, signed>, !polang.integer<64, signed> -> !polang.integer<64, signed>
// CHECK-NEXT:     polang_ast.return %2 : !polang.integer<64, signed>
// CHECK-NEXT:   }
polang_ast.func @add_integers() -> !polang.integer<64, signed> {
  %0 = polang_ast.constant.integer 1 : !polang.integer<64, signed>
  %1 = polang_ast.constant.integer 2 : !polang.integer<64, signed>
  %2 = polang_ast.add %0, %1 : !polang.integer<64, signed>, !polang.integer<64, signed> -> !polang.integer<64, signed>
  polang_ast.return %2 : !polang.integer<64, signed>
}

// Type variables should be allowed - inference handles them
// CHECK-NEXT:   polang_ast.func @add_typevars() -> !polang_ast.typevar<1, integer> {
// CHECK-NEXT:     %0 = polang_ast.constant.integer 1 : !polang_ast.typevar<1, integer>
// CHECK-NEXT:     %1 = polang_ast.constant.integer 2 : !polang_ast.typevar<1, integer>
// CHECK-NEXT:     %2 = polang_ast.add %0, %1 : !polang_ast.typevar<1, integer>, !polang_ast.typevar<1, integer> -> !polang_ast.typevar<1, integer>
// CHECK-NEXT:     polang_ast.return %2 : !polang_ast.typevar<1, integer>
// CHECK-NEXT:   }
polang_ast.func @add_typevars() -> !polang_ast.typevar<1, integer> {
  %0 = polang_ast.constant.integer 1 : !polang_ast.typevar<1, integer>
  %1 = polang_ast.constant.integer 2 : !polang_ast.typevar<1, integer>
  %2 = polang_ast.add %0, %1 : !polang_ast.typevar<1, integer>, !polang_ast.typevar<1, integer> -> !polang_ast.typevar<1, integer>
  polang_ast.return %2 : !polang_ast.typevar<1, integer>
}

// CHECK-NEXT: }
