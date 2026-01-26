// RUN: %polang_opt %s -polang-resolve-names | %FileCheck %s

// Test resolving variable references inside let expression bodies

// CHECK: module {

// CHECK-NEXT:   polang_ast.func @let_single() -> !polang.integer<64, signed> {
// CHECK-NEXT:     %0 = polang_ast.constant.integer 10 : !polang.integer<64, signed>
// CHECK-NEXT:     %1 = polang_ast.let_expr ["x"] -> !polang.integer<64, signed>
// CHECK-NEXT:   binding {
// CHECK-NEXT:       polang_ast.yield.binding %0 : !polang.integer<64, signed>
// CHECK-NEXT:     }
// CHECK-NEXT:   body {
// CHECK-NEXT:     ^bb0(%arg0: !polang.integer<64, signed>):
// The var_ref "x" should be replaced with block argument %arg0
// CHECK-NEXT:       polang_ast.yield %arg0 : !polang.integer<64, signed>
// CHECK-NEXT:     }
// CHECK-NEXT:     polang_ast.return %1 : !polang.integer<64, signed>
// CHECK-NEXT:   }
polang_ast.func @let_single() -> !polang.integer<64, signed> {
  %c10 = polang_ast.constant.integer 10 : !polang.integer<64, signed>
  %result = polang_ast.let_expr ["x"] -> !polang.integer<64, signed>
    binding {
      polang_ast.yield.binding %c10 : !polang.integer<64, signed>
    }
    body {
    ^bb0(%x_arg: !polang.integer<64, signed>):
      %0 = polang_ast.var_ref "x" : !polang.integer<64, signed>
      polang_ast.yield %0 : !polang.integer<64, signed>
    }
  polang_ast.return %result : !polang.integer<64, signed>
}

// CHECK-NEXT:   polang_ast.func @let_multiple() -> !polang.integer<64, signed> {
// CHECK-NEXT:     %0 = polang_ast.constant.integer 1 : !polang.integer<64, signed>
// CHECK-NEXT:     %1 = polang_ast.constant.integer 2 : !polang.integer<64, signed>
// CHECK-NEXT:     %2 = polang_ast.let_expr ["a", "b"] -> !polang.integer<64, signed>
// CHECK-NEXT:   binding {
// CHECK-NEXT:       polang_ast.yield.binding %0 : !polang.integer<64, signed>
// CHECK-NEXT:     }
// CHECK-NEXT:   binding {
// CHECK-NEXT:       polang_ast.yield.binding %1 : !polang.integer<64, signed>
// CHECK-NEXT:     }
// CHECK-NEXT:   body {
// CHECK-NEXT:     ^bb0(%arg0: !polang.integer<64, signed>, %arg1: !polang.integer<64, signed>):
// The var_refs "a" and "b" should be replaced with block arguments %arg0 and %arg1
// CHECK-NEXT:       %3 = polang_ast.add %arg0, %arg1 : !polang.integer<64, signed>, !polang.integer<64, signed> -> !polang.integer<64, signed>
// CHECK-NEXT:       polang_ast.yield %3 : !polang.integer<64, signed>
// CHECK-NEXT:     }
// CHECK-NEXT:     polang_ast.return %2 : !polang.integer<64, signed>
// CHECK-NEXT:   }

polang_ast.func @let_multiple() -> !polang.integer<64, signed> {
  %c1 = polang_ast.constant.integer 1 : !polang.integer<64, signed>
  %c2 = polang_ast.constant.integer 2 : !polang.integer<64, signed>
  %result = polang_ast.let_expr ["a", "b"] -> !polang.integer<64, signed>
    binding {
      polang_ast.yield.binding %c1 : !polang.integer<64, signed>
    }
    binding {
      polang_ast.yield.binding %c2 : !polang.integer<64, signed>
    }
    body {
    ^bb0(%a_arg: !polang.integer<64, signed>, %b_arg: !polang.integer<64, signed>):
      %0 = polang_ast.var_ref "a" : !polang.integer<64, signed>
      %1 = polang_ast.var_ref "b" : !polang.integer<64, signed>
      %2 = polang_ast.add %0, %1 : !polang.integer<64, signed>, !polang.integer<64, signed> -> !polang.integer<64, signed>
      polang_ast.yield %2 : !polang.integer<64, signed>
    }
  polang_ast.return %result : !polang.integer<64, signed>
}

// CHECK-NEXT: }
