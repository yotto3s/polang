// RUN: %not %polang_opt %s -polang-type-check 2>&1 | %FileCheck %s

// Test that type mismatches in binary operations produce errors
// Note: The AddOp verifier catches that bool isn't a valid numeric operand

// CHECK: {{.*}}error: 'polang_ast.add' op operand #1 must be Polang numeric type or type variable, but got '!polang.bool'
polang_ast.func @add_int_bool() -> !polang.integer<64, signed> {
  %0 = polang_ast.constant.integer 1 : !polang.integer<64, signed>
  %1 = polang_ast.constant.bool true : !polang.bool
  // Trying to add integer and bool should fail
  %2 = polang_ast.add %0, %1 : !polang.integer<64, signed>, !polang.bool -> !polang.integer<64, signed>
  polang_ast.return %2 : !polang.integer<64, signed>
}
