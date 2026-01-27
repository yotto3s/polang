// RUN: %polang_opt %s -split-input-file -verify-diagnostics

// =============================================================================
// Test: Undeclared type parameter in function signature
// =============================================================================

// expected-error @below {{use of undeclared type parameter 'U'}}
polang.generic_func @undeclared_param<T>(%arg0: !polang.param<"U">) -> !polang.param<"T"> {
  polang.return %arg0 : !polang.param<"U">
}

// -----

// =============================================================================
// Test: Duplicate type parameter names
// =============================================================================

// expected-error @below {{duplicate type parameter name 'T'}}
polang.generic_func @duplicate_param<T, T>(%arg0: !polang.param<"T">) -> !polang.param<"T"> {
  polang.return %arg0 : !polang.param<"T">
}

// -----

// =============================================================================
// Test: Return type uses undeclared type parameter
// =============================================================================

// expected-error @below {{use of undeclared type parameter 'X'}}
polang.generic_func @undeclared_return<T>(%arg0: !polang.param<"T">) -> !polang.param<"X"> {
  polang.return %arg0 : !polang.param<"T">
}
