// RUN: %not %polang_opt %s 2>&1 | %FileCheck %s

// Test that if condition must be bool or unconstrained type variable.
// Type variables with 'integer' or 'float' kind are NOT compatible with bool.

// CHECK: {{.*}}error: 'polang_ast.if' op condition type must be bool or unconstrained type variable, got '!polang_ast.typevar<1, integer>'
%0 = polang_ast.constant.integer 1 : !polang_ast.typevar<1, integer>
%1 = polang_ast.constant.integer 10 : !polang.integer<64, signed>
%2 = polang_ast.if %0 : !polang_ast.typevar<1, integer> -> !polang.integer<64, signed> {
  polang_ast.yield %1 : !polang.integer<64, signed>
} else {
  polang_ast.yield %1 : !polang.integer<64, signed>
}
