// RUN: %polang_opt %s | %polang_opt | %FileCheck %s

// Test polang_ast function with captures attribute

// CHECK: module {

// Test function without captures (default)
// CHECK-NEXT:   polang_ast.func @no_captures(%arg0: !polang.integer<64, signed>) -> !polang.integer<64, signed> {
// CHECK-NEXT:     polang_ast.return %arg0 : !polang.integer<64, signed>
// CHECK-NEXT:   }
polang_ast.func @no_captures(%x: !polang.integer<64, signed>) -> !polang.integer<64, signed> {
  polang_ast.return %x : !polang.integer<64, signed>
}

// Test function with single capture
// CHECK-NEXT:   polang_ast.func @single_capture(%arg0: !polang.integer<64, signed>) -> !polang.integer<64, signed> attributes {captures = ["outer"]} {
// CHECK-NEXT:     polang_ast.return %arg0 : !polang.integer<64, signed>
// CHECK-NEXT:   }
polang_ast.func @single_capture(%x: !polang.integer<64, signed>) -> !polang.integer<64, signed> attributes {captures = ["outer"]} {
  polang_ast.return %x : !polang.integer<64, signed>
}

// Test function with multiple captures
// CHECK-NEXT:   polang_ast.func @multiple_captures(%arg0: !polang.integer<64, signed>) -> !polang.integer<64, signed> attributes {captures = ["a", "b", "c"]} {
// CHECK-NEXT:     polang_ast.return %arg0 : !polang.integer<64, signed>
// CHECK-NEXT:   }
polang_ast.func @multiple_captures(%x: !polang.integer<64, signed>) -> !polang.integer<64, signed> attributes {captures = ["a", "b", "c"]} {
  polang_ast.return %x : !polang.integer<64, signed>
}

// Test closure-like function with captures and type variables
// CHECK-NEXT:   polang_ast.func @closure_with_typevar(%arg0: !polang_ast.typevar<1>) -> !polang_ast.typevar<1> attributes {captures = ["x", "y"]} {
// CHECK-NEXT:     polang_ast.return %arg0 : !polang_ast.typevar<1>
// CHECK-NEXT:   }
polang_ast.func @closure_with_typevar(%a: !polang_ast.typevar<1>) -> !polang_ast.typevar<1> attributes {captures = ["x", "y"]} {
  polang_ast.return %a : !polang_ast.typevar<1>
}

// CHECK-NEXT: }
