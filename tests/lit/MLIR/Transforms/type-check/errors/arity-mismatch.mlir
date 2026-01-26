// RUN: %not %polang_opt %s -polang-type-check 2>&1 | %FileCheck %s

// Test that function call arity mismatches produce errors
// Note: The CallOp verifier already checks arity, producing this error

polang_ast.func @add(%arg0: !polang.integer<64, signed>, %arg1: !polang.integer<64, signed>) -> !polang.integer<64, signed> {
  %0 = polang_ast.add %arg0, %arg1 : !polang.integer<64, signed>, !polang.integer<64, signed> -> !polang.integer<64, signed>
  polang_ast.return %0 : !polang.integer<64, signed>
}

// CHECK: {{.*}}error: 'polang_ast.call' op incorrect number of operands for callee
polang_ast.func @caller() -> !polang.integer<64, signed> {
  %0 = polang_ast.constant.integer 1 : !polang.integer<64, signed>
  // Calling add with 1 argument instead of 2 should fail
  %1 = polang_ast.call @add(%0) : (!polang.integer<64, signed>) -> !polang.integer<64, signed>
  polang_ast.return %1 : !polang.integer<64, signed>
}
