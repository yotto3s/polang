// RUN: %polang_opt %s -split-input-file -verify-diagnostics

// Note: "binding count must match var_names count" is enforced at parse time,
// not at verifier time, since the parser reads exactly var_names.size() bindings.

// Test that body block argument count must match binding count
%0 = polang.constant.integer 10 : !polang.integer<64, signed>
%1 = polang.constant.integer 20 : !polang.integer<64, signed>

// expected-error @+1 {{body block argument count (1) doesn't match binding count (2)}}
%2 = polang.let_expr ["x", "y"] -> !polang.integer<64, signed>
  binding {
    polang.yield.binding %0 : !polang.integer<64, signed>
  }
  binding {
    polang.yield.binding %1 : !polang.integer<64, signed>
  }
  body {
  ^bb0(%x_arg: !polang.integer<64, signed>):  // Only one arg, but two bindings
    polang.yield %x_arg : !polang.integer<64, signed>
  }

// -----

// Test that yield type must match result type
%0 = polang.constant.integer 10 : !polang.integer<64, signed>
%1 = polang.constant.float 3.14 : !polang.float<64>

// expected-error @+1 {{yield type '!polang.float<64>' doesn't match result type '!polang.integer<64, signed>'}}
%2 = polang.let_expr ["x"] -> !polang.integer<64, signed>
  binding {
    polang.yield.binding %0 : !polang.integer<64, signed>
  }
  body {
  ^bb0(%x_arg: !polang.integer<64, signed>):
    polang.yield %1 : !polang.float<64>
  }

// -----

// Test that binding region must end with yield.binding
%0 = polang.constant.integer 10 : !polang.integer<64, signed>

// expected-error @+1 {{binding region #0 must end with polang.yield.binding}}
%1 = polang.let_expr ["x"] -> !polang.integer<64, signed>
  binding {
    polang.yield %0 : !polang.integer<64, signed>
  }
  body {
  ^bb0(%x_arg: !polang.integer<64, signed>):
    polang.yield %x_arg : !polang.integer<64, signed>
  }
