// RUN: %polang_opt %s | %polang_opt | %FileCheck %s

// Test polang_ast module operations

// CHECK: module {

// Test module declaration with import
// CHECK-NEXT:   polang_ast.module @Math {
// CHECK-NEXT:     polang_ast.func @add(%arg0: !polang.integer<64, signed>, %arg1: !polang.integer<64, signed>) -> !polang.integer<64, signed> {
// CHECK-NEXT:       %0 = polang_ast.add %arg0, %arg1 : !polang.integer<64, signed>, !polang.integer<64, signed> -> !polang.integer<64, signed>
// CHECK-NEXT:       polang_ast.return %0 : !polang.integer<64, signed>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
polang_ast.module @Math {
  polang_ast.func @add(%a: !polang.integer<64, signed>, %b: !polang.integer<64, signed>) -> !polang.integer<64, signed> {
    %0 = polang_ast.add %a, %b : !polang.integer<64, signed>, !polang.integer<64, signed> -> !polang.integer<64, signed>
    polang_ast.return %0 : !polang.integer<64, signed>
  }
}

// Test import statement
// CHECK-NEXT:   polang_ast.import Math
polang_ast.import @Math

// Test import with alias
// CHECK-NEXT:   polang_ast.import Math as M
polang_ast.import @Math as @M

// CHECK-NEXT: }
