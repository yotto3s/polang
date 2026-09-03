// RUN: %polang_opt %s -split-input-file -verify-diagnostics

// Note: "binding count must match var_names count" is guaranteed for textual IR
// because the parser reads exactly var_names.size() bindings; the verifier still
// enforces this invariant for programmatically constructed operations.
// Test that body block argument count must match binding count
%0 = polang.constant.integer 10 : si64
%1 = polang.constant.integer 20 : si64

// expected-error @+1 {{body block argument count (1) doesn't match binding count (2)}}
%2 = polang.let_expr ["x", "y"] -> si64
  binding {
    polang.yield.binding %0 : si64
  }
  binding {
    polang.yield.binding %1 : si64
  }
  body {
  ^bb0(%x_arg: si64):  // Only one arg, but two bindings
    polang.yield %x_arg : si64
  }

// -----

// Test that yield type must match result type
%0 = polang.constant.integer 10 : si64
%1 = polang.constant.float 3.14 : f64

// expected-error @+1 {{yield type 'f64' doesn't match result type 'si64'}}
%2 = polang.let_expr ["x"] -> si64
  binding {
    polang.yield.binding %0 : si64
  }
  body {
  ^bb0(%x_arg: si64):
    polang.yield %1 : f64
  }

// -----

// Test that binding region must end with yield.binding
%0 = polang.constant.integer 10 : si64

// expected-error @+1 {{binding region #0 must end with polang.yield.binding}}
%1 = polang.let_expr ["x"] -> si64
  binding {
    polang.yield %0 : si64
  }
  body {
  ^bb0(%x_arg: si64):
    polang.yield %x_arg : si64
  }
