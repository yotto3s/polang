// RUN: %polang_opt %s | %polang_opt | %FileCheck %s

// Test polang_ast let expression operations

// CHECK: module {

// Test let expression with single binding
// CHECK-NEXT:   %0 = polang_ast.constant.integer 10 : !polang.integer<64, signed>
%0 = polang_ast.constant.integer 10 : !polang.integer<64, signed>
// CHECK-NEXT:   %1 = polang_ast.constant.integer 20 : !polang.integer<64, signed>
%1 = polang_ast.constant.integer 20 : !polang.integer<64, signed>
// CHECK-NEXT:   %2 = polang_ast.let_expr -> !polang.integer<64, signed> {
// CHECK-NEXT:     polang_ast.var_bind "x" = %0 : !polang.integer<64, signed>
// CHECK-NEXT:     polang_ast.var_bind "y" = %1 : !polang.integer<64, signed>
// CHECK-NEXT:     %3 = polang_ast.add %0, %1 : !polang.integer<64, signed>, !polang.integer<64, signed> -> !polang.integer<64, signed>
// CHECK-NEXT:     polang_ast.yield %3 : !polang.integer<64, signed>
// CHECK-NEXT:   }
%2 = polang_ast.let_expr -> !polang.integer<64, signed> {
  polang_ast.var_bind "x" = %0 : !polang.integer<64, signed>
  polang_ast.var_bind "y" = %1 : !polang.integer<64, signed>
  %3 = polang_ast.add %0, %1 : !polang.integer<64, signed>, !polang.integer<64, signed> -> !polang.integer<64, signed>
  polang_ast.yield %3 : !polang.integer<64, signed>
}

// CHECK-NEXT: }
