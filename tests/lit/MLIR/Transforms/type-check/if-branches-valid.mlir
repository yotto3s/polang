// RUN: %polang_opt %s -polang-type-check | %FileCheck %s

// Test that valid if expressions pass type checking

// CHECK: module {

// CHECK-NEXT:   polang_ast.func @if_with_bool_condition() -> !polang.integer<64, signed> {
// CHECK-NEXT:     %0 = polang_ast.constant.bool true : !polang.bool
// CHECK-NEXT:     %1 = polang_ast.constant.integer 1 : !polang.integer<64, signed>
// CHECK-NEXT:     %2 = polang_ast.constant.integer 2 : !polang.integer<64, signed>
// CHECK-NEXT:     %3 = polang_ast.if %0 : !polang.bool -> !polang.integer<64, signed> {
// CHECK-NEXT:       polang_ast.yield %1 : !polang.integer<64, signed>
// CHECK-NEXT:     } else {
// CHECK-NEXT:       polang_ast.yield %2 : !polang.integer<64, signed>
// CHECK-NEXT:     }
// CHECK-NEXT:     polang_ast.return %3 : !polang.integer<64, signed>
// CHECK-NEXT:   }
polang_ast.func @if_with_bool_condition() -> !polang.integer<64, signed> {
  %cond = polang_ast.constant.bool true : !polang.bool
  %then = polang_ast.constant.integer 1 : !polang.integer<64, signed>
  %else = polang_ast.constant.integer 2 : !polang.integer<64, signed>
  %result = polang_ast.if %cond : !polang.bool -> !polang.integer<64, signed> {
    polang_ast.yield %then : !polang.integer<64, signed>
  } else {
    polang_ast.yield %else : !polang.integer<64, signed>
  }
  polang_ast.return %result : !polang.integer<64, signed>
}

// Test function with typevar parameter - the "any" kind typevar is allowed as condition
// Note: "any" is the default kind so it's not printed
// CHECK-NEXT:   polang_ast.func @if_with_typevar_condition(%arg0: !polang_ast.typevar<1>) -> !polang.integer<64, signed> {
// CHECK-NEXT:     %0 = polang_ast.constant.integer 1 : !polang.integer<64, signed>
// CHECK-NEXT:     %1 = polang_ast.constant.integer 2 : !polang.integer<64, signed>
// CHECK-NEXT:     %2 = polang_ast.if %arg0 : !polang_ast.typevar<1> -> !polang.integer<64, signed> {
// CHECK-NEXT:       polang_ast.yield %0 : !polang.integer<64, signed>
// CHECK-NEXT:     } else {
// CHECK-NEXT:       polang_ast.yield %1 : !polang.integer<64, signed>
// CHECK-NEXT:     }
// CHECK-NEXT:     polang_ast.return %2 : !polang.integer<64, signed>
// CHECK-NEXT:   }
polang_ast.func @if_with_typevar_condition(%cond: !polang_ast.typevar<1, any>) -> !polang.integer<64, signed> {
  %then = polang_ast.constant.integer 1 : !polang.integer<64, signed>
  %else = polang_ast.constant.integer 2 : !polang.integer<64, signed>
  %result = polang_ast.if %cond : !polang_ast.typevar<1, any> -> !polang.integer<64, signed> {
    polang_ast.yield %then : !polang.integer<64, signed>
  } else {
    polang_ast.yield %else : !polang.integer<64, signed>
  }
  polang_ast.return %result : !polang.integer<64, signed>
}

// CHECK-NEXT: }
