// RUN: %not %polang_opt %s 2>&1 | %FileCheck %s

// Test that if condition rejects float kind type variable

// CHECK: {{.*}}error: 'polang_ast.if' op condition type must be bool or unconstrained type variable, got '!polang_ast.typevar<1, float>'
%0 = polang_ast.constant.float 1.0 : !polang_ast.typevar<1, float>
%1 = polang_ast.constant.integer 10 : !polang.integer<64, signed>
%2 = polang_ast.if %0 : !polang_ast.typevar<1, float> -> !polang.integer<64, signed> {
  polang_ast.yield %1 : !polang.integer<64, signed>
} else {
  polang_ast.yield %1 : !polang.integer<64, signed>
}
