// RUN: %polang_opt %s -split-input-file -verify-diagnostics

// Note: in textual form, "binding count must match var_names count" is guaranteed
// by the parser (it reads exactly var_names.size() bindings), but the verifier
// still enforces this for programmatic IR construction.

// Test that body block argument count must match binding count
%0 = polang_ast.constant.integer 10 : !polang.integer<64, signed>
%1 = polang_ast.constant.integer 20 : !polang.integer<64, signed>

// expected-error @+1 {{body block argument count (1) doesn't match binding count (2)}}
%2 = polang_ast.let_expr ["x", "y"] -> !polang.integer<64, signed>
  binding {
    polang_ast.yield.binding %0 : !polang.integer<64, signed>
  }
  binding {
    polang_ast.yield.binding %1 : !polang.integer<64, signed>
  }
  body {
  ^bb0(%x_arg: !polang.integer<64, signed>):  // Only one arg, but two bindings
    polang_ast.yield %x_arg : !polang.integer<64, signed>
  }

// -----

// Test that yield type must match result type
%0 = polang_ast.constant.integer 10 : !polang.integer<64, signed>
%1 = polang_ast.constant.float 3.14 : !polang.float<64>

// expected-error @+1 {{yield type '!polang.float<64>' doesn't match result type '!polang.integer<64, signed>'}}
%2 = polang_ast.let_expr ["x"] -> !polang.integer<64, signed>
  binding {
    polang_ast.yield.binding %0 : !polang.integer<64, signed>
  }
  body {
  ^bb0(%x_arg: !polang.integer<64, signed>):
    polang_ast.yield %1 : !polang.float<64>
  }

// -----

// Test that binding region must end with yield.binding
%0 = polang_ast.constant.integer 10 : !polang.integer<64, signed>

// expected-error @+1 {{binding region #0 must end with polang_ast.yield.binding}}
%1 = polang_ast.let_expr ["x"] -> !polang.integer<64, signed>
  binding {
    polang_ast.yield %0 : !polang.integer<64, signed>
  }
  body {
  ^bb0(%x_arg: !polang.integer<64, signed>):
    polang_ast.yield %x_arg : !polang.integer<64, signed>
  }
