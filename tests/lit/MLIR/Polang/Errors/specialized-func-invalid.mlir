// RUN: %polang_opt %s -split-input-file -verify-diagnostics

// =============================================================================
// Setup: Generic functions to reference
// =============================================================================

polang.generic_func @identity<T>(%arg0: !polang.param<"T">) -> !polang.param<"T"> {
  polang.return %arg0 : !polang.param<"T">
}

polang.generic_func @add<T: numeric>(%arg0: !polang.param<"T", numeric>, %arg1: !polang.param<"T", numeric>) -> !polang.param<"T", numeric> {
  %r = polang.add %arg0, %arg1 : !polang.param<"T", numeric>, !polang.param<"T", numeric> -> !polang.param<"T", numeric>
  polang.return %r : !polang.param<"T", numeric>
}

polang.generic_func @pair<A, B>(%arg0: !polang.param<"A">, %arg1: !polang.param<"B">) -> !polang.param<"A"> {
  polang.return %arg0 : !polang.param<"A">
}

// =============================================================================
// Test: Reference to non-existent generic function
// =============================================================================

// expected-error @below {{references undefined generic function 'nonexistent'}}
polang.specialized_func @bad_ref from @nonexistent<[!polang.integer<64, signed>]>

// -----

polang.generic_func @identity<T>(%arg0: !polang.param<"T">) -> !polang.param<"T"> {
  polang.return %arg0 : !polang.param<"T">
}

// =============================================================================
// Test: Wrong number of type arguments (too many)
// =============================================================================

// expected-error @below {{expects 1 type argument(s) but got 2}}
polang.specialized_func @too_many from @identity<[!polang.integer<64, signed>, !polang.float<64>]>

// -----

polang.generic_func @pair<A, B>(%arg0: !polang.param<"A">, %arg1: !polang.param<"B">) -> !polang.param<"A"> {
  polang.return %arg0 : !polang.param<"A">
}

// =============================================================================
// Test: Wrong number of type arguments (too few)
// =============================================================================

// expected-error @below {{expects 2 type argument(s) but got 1}}
polang.specialized_func @too_few from @pair<[!polang.integer<64, signed>]>

// -----

polang.generic_func @add<T: numeric>(%arg0: !polang.param<"T", numeric>, %arg1: !polang.param<"T", numeric>) -> !polang.param<"T", numeric> {
  %r = polang.add %arg0, %arg1 : !polang.param<"T", numeric>, !polang.param<"T", numeric> -> !polang.param<"T", numeric>
  polang.return %r : !polang.param<"T", numeric>
}

// =============================================================================
// Test: Type argument violates constraint (bool is not numeric)
// =============================================================================

// expected-error @below {{type argument '!polang.bool' does not satisfy 'numeric' constraint for type parameter 'T'}}
polang.specialized_func @add_bool from @add<[!polang.bool]>

// -----

polang.generic_func @int_only<T: integer>(%arg0: !polang.param<"T", integer>) -> !polang.param<"T", integer> {
  polang.return %arg0 : !polang.param<"T", integer>
}

// =============================================================================
// Test: Type argument violates integer constraint (float given)
// =============================================================================

// expected-error @below {{type argument '!polang.float<64>' does not satisfy 'integer' constraint for type parameter 'T'}}
polang.specialized_func @int_with_float from @int_only<[!polang.float<64>]>

// -----

polang.generic_func @float_only<T: float>(%arg0: !polang.param<"T", float>) -> !polang.param<"T", float> {
  polang.return %arg0 : !polang.param<"T", float>
}

// =============================================================================
// Test: Type argument violates float constraint (integer given)
// =============================================================================

// expected-error @below {{type argument '!polang.integer<64, signed>' does not satisfy 'float' constraint for type parameter 'T'}}
polang.specialized_func @float_with_int from @float_only<[!polang.integer<64, signed>]>
