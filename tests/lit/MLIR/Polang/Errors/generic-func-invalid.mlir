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
// Test: Empty type parameter list (not allowed)
// =============================================================================

// expected-error @below {{generic function must have at least one type parameter}}
polang.generic_func @empty_params<>(%arg0: !polang.integer<64, signed>) -> !polang.integer<64, signed> {
  polang.return %arg0 : !polang.integer<64, signed>
}

// -----

// =============================================================================
// Test: Return type uses undeclared type parameter
// =============================================================================

// expected-error @below {{use of undeclared type parameter 'X'}}
polang.generic_func @undeclared_return<T>(%arg0: !polang.param<"T">) -> !polang.param<"X"> {
  polang.return %arg0 : !polang.param<"T">
}
