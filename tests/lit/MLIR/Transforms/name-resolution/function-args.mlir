// RUN: %polang_opt %s -polang-resolve-names | %FileCheck %s

// Test resolving variable references to function arguments
// Note: Function argument names must be stored via arg_attrs with "polang.name"

// CHECK: module {

// CHECK-NEXT:   polang_ast.func @identity(%arg0: !polang.integer<64, signed> {polang.name = "x"}) -> !polang.integer<64, signed> {
// The var_ref "x" should be replaced with the function argument %arg0
// CHECK-NEXT:     polang_ast.return %arg0 : !polang.integer<64, signed>
// CHECK-NEXT:   }
polang_ast.func @identity(%x: !polang.integer<64, signed> {polang.name = "x"}) -> !polang.integer<64, signed> {
  // The var_ref "x" should be replaced with the function argument
  %0 = polang_ast.var_ref "x" : !polang.integer<64, signed>
  polang_ast.return %0 : !polang.integer<64, signed>
}

// CHECK-NEXT:   polang_ast.func @add(%arg0: !polang.integer<64, signed> {polang.name = "a"}, %arg1: !polang.integer<64, signed> {polang.name = "b"}) -> !polang.integer<64, signed> {
// CHECK-NEXT:     %0 = polang_ast.add %arg0, %arg1 : !polang.integer<64, signed>, !polang.integer<64, signed> -> !polang.integer<64, signed>
// CHECK-NEXT:     polang_ast.return %0 : !polang.integer<64, signed>
// CHECK-NEXT:   }
polang_ast.func @add(%a: !polang.integer<64, signed> {polang.name = "a"}, %b: !polang.integer<64, signed> {polang.name = "b"}) -> !polang.integer<64, signed> {
  %0 = polang_ast.var_ref "a" : !polang.integer<64, signed>
  %1 = polang_ast.var_ref "b" : !polang.integer<64, signed>
  %2 = polang_ast.add %0, %1 : !polang.integer<64, signed>, !polang.integer<64, signed> -> !polang.integer<64, signed>
  polang_ast.return %2 : !polang.integer<64, signed>
}

// CHECK-NEXT: }
