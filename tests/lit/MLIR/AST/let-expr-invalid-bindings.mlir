// RUN: %polang_opt %s -split-input-file -verify-diagnostics

// Test that non-var_bind operations in bindings region are rejected
%0 = polang_ast.constant.integer 10 : !polang.integer<64, signed>

// expected-error @+1 {{bindings region must only contain polang_ast.var_bind operations}}
%1 = polang_ast.let_expr -> !polang.integer<64, signed> {
  %x = polang_ast.var_bind "x" = %0 : !polang.integer<64, signed>
  %bad = polang_ast.constant.integer 99 : !polang.integer<64, signed>
  polang_ast.yield.bindings %x : !polang.integer<64, signed>
} do {
^bb0(%x_arg: !polang.integer<64, signed>):
  polang_ast.yield %x_arg : !polang.integer<64, signed>
}

// -----

// Test that body block argument count must match yield.bindings operand count
%0 = polang_ast.constant.integer 10 : !polang.integer<64, signed>
%1 = polang_ast.constant.integer 20 : !polang.integer<64, signed>

// expected-error @+1 {{body block argument count (1) doesn't match yield.bindings operand count (2)}}
%2 = polang_ast.let_expr -> !polang.integer<64, signed> {
  %x = polang_ast.var_bind "x" = %0 : !polang.integer<64, signed>
  %y = polang_ast.var_bind "y" = %1 : !polang.integer<64, signed>
  polang_ast.yield.bindings %x, %y : !polang.integer<64, signed>, !polang.integer<64, signed>
} do {
^bb0(%x_arg: !polang.integer<64, signed>):  // Only one arg, but two yielded
  polang_ast.yield %x_arg : !polang.integer<64, signed>
}

// -----

// Test that yield type must match result type
%0 = polang_ast.constant.integer 10 : !polang.integer<64, signed>
%1 = polang_ast.constant.float 3.14 : !polang.float<64>

// expected-error @+1 {{yield type '!polang.float<64>' doesn't match result type '!polang.integer<64, signed>'}}
%2 = polang_ast.let_expr -> !polang.integer<64, signed> {
  %x = polang_ast.var_bind "x" = %0 : !polang.integer<64, signed>
  polang_ast.yield.bindings %x : !polang.integer<64, signed>
} do {
^bb0(%x_arg: !polang.integer<64, signed>):
  polang_ast.yield %1 : !polang.float<64>
}
