// RUN: %polang_opt %s -polang-resolve-names | %FileCheck %s

// Test resolving variables from outer scope (function parameters inside let body)

// CHECK: module {

// CHECK-NEXT:   polang_ast.func @outer() -> !polang.integer<64, signed> {
// CHECK-NEXT:     %0 = polang_ast.constant.integer 10 : !polang.integer<64, signed>
// CHECK-NEXT:     %1 = polang_ast.let_expr ["x"] -> !polang.integer<64, signed>
// CHECK-NEXT:   binding {
// CHECK-NEXT:       polang_ast.yield.binding %0 : !polang.integer<64, signed>
// CHECK-NEXT:     }
// CHECK-NEXT:   body {
// CHECK-NEXT:     ^bb0(%arg0: !polang.integer<64, signed>):
// Reference x from the let binding - this should resolve to block arg
// CHECK-NEXT:       polang_ast.yield %arg0 : !polang.integer<64, signed>
// CHECK-NEXT:     }
// CHECK-NEXT:     polang_ast.return %1 : !polang.integer<64, signed>
// CHECK-NEXT:   }
polang_ast.func @outer() -> !polang.integer<64, signed> {
  %c10 = polang_ast.constant.integer 10 : !polang.integer<64, signed>
  %result = polang_ast.let_expr ["x"] -> !polang.integer<64, signed>
    binding {
      polang_ast.yield.binding %c10 : !polang.integer<64, signed>
    }
    body {
    ^bb0(%x_arg: !polang.integer<64, signed>):
      // Reference x from the let binding - this should resolve to block arg
      %0 = polang_ast.var_ref "x" : !polang.integer<64, signed>
      polang_ast.yield %0 : !polang.integer<64, signed>
    }
  polang_ast.return %result : !polang.integer<64, signed>
}

// Test: function argument access within nested let
// CHECK-NEXT:   polang_ast.func @with_param(%arg0: !polang.integer<64, signed> {polang.name = "y"}) -> !polang.integer<64, signed> {
// CHECK-NEXT:     %0 = polang_ast.constant.integer 5 : !polang.integer<64, signed>
// CHECK-NEXT:     %1 = polang_ast.let_expr ["x"] -> !polang.integer<64, signed>
// CHECK-NEXT:   binding {
// CHECK-NEXT:       polang_ast.yield.binding %0 : !polang.integer<64, signed>
// CHECK-NEXT:     }
// CHECK-NEXT:   body {
// CHECK-NEXT:     ^bb0(%arg1: !polang.integer<64, signed>):
// x from let -> %arg1, y from function param -> %arg0
// CHECK-NEXT:       %2 = polang_ast.add %arg1, %arg0 : !polang.integer<64, signed>, !polang.integer<64, signed> -> !polang.integer<64, signed>
// CHECK-NEXT:       polang_ast.yield %2 : !polang.integer<64, signed>
// CHECK-NEXT:     }
// CHECK-NEXT:     polang_ast.return %1 : !polang.integer<64, signed>
// CHECK-NEXT:   }
polang_ast.func @with_param(%y: !polang.integer<64, signed> {polang.name = "y"}) -> !polang.integer<64, signed> {
  %c5 = polang_ast.constant.integer 5 : !polang.integer<64, signed>
  %result = polang_ast.let_expr ["x"] -> !polang.integer<64, signed>
    binding {
      polang_ast.yield.binding %c5 : !polang.integer<64, signed>
    }
    body {
    ^bb0(%x_arg: !polang.integer<64, signed>):
      // Reference both x (from let) and y (from function param)
      %0 = polang_ast.var_ref "x" : !polang.integer<64, signed>
      // y comes from outer scope - the function parameter
      %1 = polang_ast.var_ref "y" : !polang.integer<64, signed>
      %2 = polang_ast.add %0, %1 : !polang.integer<64, signed>, !polang.integer<64, signed> -> !polang.integer<64, signed>
      polang_ast.yield %2 : !polang.integer<64, signed>
    }
  polang_ast.return %result : !polang.integer<64, signed>
}

// CHECK-NEXT: }
