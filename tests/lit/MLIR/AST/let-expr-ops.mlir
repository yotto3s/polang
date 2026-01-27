// RUN: %polang_opt %s | %polang_opt | %FileCheck %s

// Test polang_ast let expression operations with variadic binding regions

// CHECK: module {

// Test let expression with two bindings
// CHECK-NEXT:   %[[C10:.*]] = polang_ast.constant.integer 10 : !polang.integer<64, signed>
%0 = polang_ast.constant.integer 10 : !polang.integer<64, signed>
// CHECK-NEXT:   %[[C20:.*]] = polang_ast.constant.integer 20 : !polang.integer<64, signed>
%1 = polang_ast.constant.integer 20 : !polang.integer<64, signed>
// CHECK-NEXT:   %[[RESULT:.*]] = polang_ast.let_expr ["x", "y"] -> !polang.integer<64, signed>
// CHECK-NEXT:     binding {
// CHECK-NEXT:       polang_ast.yield.binding %[[C10]] : !polang.integer<64, signed>
// CHECK-NEXT:     }
// CHECK-NEXT:     binding {
// CHECK-NEXT:       polang_ast.yield.binding %[[C20]] : !polang.integer<64, signed>
// CHECK-NEXT:     }
// CHECK-NEXT:     body {
// CHECK-NEXT:     ^bb0(%[[X_ARG:.*]]: !polang.integer<64, signed>, %[[Y_ARG:.*]]: !polang.integer<64, signed>):
// CHECK-NEXT:       %[[SUM:.*]] = polang_ast.add %[[X_ARG]], %[[Y_ARG]] : !polang.integer<64, signed>, !polang.integer<64, signed> -> !polang.integer<64, signed>
// CHECK-NEXT:       polang_ast.yield %[[SUM]] : !polang.integer<64, signed>
// CHECK-NEXT:     }
%2 = polang_ast.let_expr ["x", "y"] -> !polang.integer<64, signed>
  binding {
    polang_ast.yield.binding %0 : !polang.integer<64, signed>
  }
  binding {
    polang_ast.yield.binding %1 : !polang.integer<64, signed>
  }
  body {
  ^bb0(%x_arg: !polang.integer<64, signed>, %y_arg: !polang.integer<64, signed>):
    %sum = polang_ast.add %x_arg, %y_arg : !polang.integer<64, signed>, !polang.integer<64, signed> -> !polang.integer<64, signed>
    polang_ast.yield %sum : !polang.integer<64, signed>
  }

// Test let expression with single binding
// CHECK:       %[[C5:.*]] = polang_ast.constant.integer 5 : !polang.integer<64, signed>
%3 = polang_ast.constant.integer 5 : !polang.integer<64, signed>
// CHECK-NEXT:   %[[SINGLE:.*]] = polang_ast.let_expr ["n"] -> !polang.integer<64, signed>
// CHECK-NEXT:     binding {
// CHECK-NEXT:       polang_ast.yield.binding %[[C5]] : !polang.integer<64, signed>
// CHECK-NEXT:     }
// CHECK-NEXT:     body {
// CHECK-NEXT:     ^bb0(%[[N_ARG:.*]]: !polang.integer<64, signed>):
// CHECK-NEXT:       polang_ast.yield %[[N_ARG]] : !polang.integer<64, signed>
// CHECK-NEXT:     }
%4 = polang_ast.let_expr ["n"] -> !polang.integer<64, signed>
  binding {
    polang_ast.yield.binding %3 : !polang.integer<64, signed>
  }
  body {
  ^bb0(%n_arg: !polang.integer<64, signed>):
    polang_ast.yield %n_arg : !polang.integer<64, signed>
  }

// CHECK-NEXT: }
