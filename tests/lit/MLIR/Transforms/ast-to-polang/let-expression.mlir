// RUN: %polang_opt %s -polang-ast-to-polang | %FileCheck %s

// Test converting let expressions - they become inline SSA values in Polang dialect

// CHECK: module {

// Let expression with single binding should be inlined
// CHECK-NEXT:   polang.func @with_let() -> !polang.integer<64, signed> {
// CHECK-NEXT:     %0 = polang.constant.integer 10 : !polang.integer<64, signed>
// The body just returns the bound value - x is inlined to %0
// CHECK-NEXT:     polang.return %0 : !polang.integer<64, signed>
// CHECK-NEXT:   }
polang_ast.func @with_let() -> !polang.integer<64, signed> {
  %c10 = polang_ast.constant.integer 10 : !polang.integer<64, signed>
  %result = polang_ast.let_expr ["x"] -> !polang.integer<64, signed>
    binding {
      polang_ast.yield.binding %c10 : !polang.integer<64, signed>
    }
    body {
    ^bb0(%x_arg: !polang.integer<64, signed>):
      polang_ast.yield %x_arg : !polang.integer<64, signed>
    }
  polang_ast.return %result : !polang.integer<64, signed>
}

// Let expression used in computation
// CHECK-NEXT:   polang.func @let_with_add() -> !polang.integer<64, signed> {
// CHECK-NEXT:     %0 = polang.constant.integer 5 : !polang.integer<64, signed>
// CHECK-NEXT:     %1 = polang.constant.integer 3 : !polang.integer<64, signed>
// CHECK-NEXT:     %2 = polang.add %0, %1 : !polang.integer<64, signed>, !polang.integer<64, signed> -> !polang.integer<64, signed>
// CHECK-NEXT:     polang.return %2 : !polang.integer<64, signed>
// CHECK-NEXT:   }
polang_ast.func @let_with_add() -> !polang.integer<64, signed> {
  %c5 = polang_ast.constant.integer 5 : !polang.integer<64, signed>
  %c3 = polang_ast.constant.integer 3 : !polang.integer<64, signed>
  %result = polang_ast.let_expr ["x", "y"] -> !polang.integer<64, signed>
    binding {
      polang_ast.yield.binding %c5 : !polang.integer<64, signed>
    }
    binding {
      polang_ast.yield.binding %c3 : !polang.integer<64, signed>
    }
    body {
    ^bb0(%x_arg: !polang.integer<64, signed>, %y_arg: !polang.integer<64, signed>):
      %sum = polang_ast.add %x_arg, %y_arg : !polang.integer<64, signed>, !polang.integer<64, signed> -> !polang.integer<64, signed>
      polang_ast.yield %sum : !polang.integer<64, signed>
    }
  polang_ast.return %result : !polang.integer<64, signed>
}

// CHECK-NEXT: }
