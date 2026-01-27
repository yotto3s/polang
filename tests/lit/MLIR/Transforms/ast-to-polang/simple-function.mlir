// RUN: %polang_opt %s -polang-ast-to-polang | %FileCheck %s

// Test converting a simple monomorphic function from AST dialect to Polang dialect

// CHECK: module {

// A simple function with concrete types should convert to polang.func
// CHECK-NEXT:   polang.func @identity(%arg0: !polang.integer<64, signed>) -> !polang.integer<64, signed> {
// CHECK-NEXT:     polang.return %arg0 : !polang.integer<64, signed>
// CHECK-NEXT:   }
polang_ast.func @identity(%arg0: !polang.integer<64, signed>) -> !polang.integer<64, signed> {
  polang_ast.return %arg0 : !polang.integer<64, signed>
}

// Function with arithmetic operations
// CHECK-NEXT:   polang.func @add(%arg0: !polang.integer<64, signed>, %arg1: !polang.integer<64, signed>) -> !polang.integer<64, signed> {
// CHECK-NEXT:     %0 = polang.add %arg0, %arg1 : !polang.integer<64, signed>, !polang.integer<64, signed> -> !polang.integer<64, signed>
// CHECK-NEXT:     polang.return %0 : !polang.integer<64, signed>
// CHECK-NEXT:   }
polang_ast.func @add(%arg0: !polang.integer<64, signed>, %arg1: !polang.integer<64, signed>) -> !polang.integer<64, signed> {
  %0 = polang_ast.add %arg0, %arg1 : !polang.integer<64, signed>, !polang.integer<64, signed> -> !polang.integer<64, signed>
  polang_ast.return %0 : !polang.integer<64, signed>
}

// CHECK-NEXT: }
