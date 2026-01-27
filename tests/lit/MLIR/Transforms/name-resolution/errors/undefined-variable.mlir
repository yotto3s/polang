// RUN: %not %polang_opt %s -polang-resolve-names 2>&1 | %FileCheck %s

// Test that undefined variable references produce errors

// CHECK: {{.*}}error: undefined variable 'undefined_name'
polang_ast.func @undefined_var() -> !polang.integer<64, signed> {
  %0 = polang_ast.var_ref "undefined_name" : !polang.integer<64, signed>
  polang_ast.return %0 : !polang.integer<64, signed>
}
