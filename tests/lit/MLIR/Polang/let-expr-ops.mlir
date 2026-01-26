// RUN: %polang_opt %s | %polang_opt | %FileCheck %s

// Test polang let expression operations with variadic binding regions

// CHECK: module {

// Test let expression with two bindings
// CHECK-NEXT:   %[[C10:.*]] = polang.constant.integer 10 : !polang.integer<64, signed>
%0 = polang.constant.integer 10 : !polang.integer<64, signed>
// CHECK-NEXT:   %[[C20:.*]] = polang.constant.integer 20 : !polang.integer<64, signed>
%1 = polang.constant.integer 20 : !polang.integer<64, signed>
// CHECK-NEXT:   %[[RESULT:.*]] = polang.let_expr ["x", "y"] -> !polang.integer<64, signed>
// CHECK-NEXT:   binding {
// CHECK-NEXT:     polang.yield.binding %[[C10]] : !polang.integer<64, signed>
// CHECK-NEXT:   }
// CHECK-NEXT:   binding {
// CHECK-NEXT:     polang.yield.binding %[[C20]] : !polang.integer<64, signed>
// CHECK-NEXT:   }
// CHECK-NEXT:   body {
// CHECK-NEXT:   ^bb0(%[[X_ARG:.*]]: !polang.integer<64, signed>, %[[Y_ARG:.*]]: !polang.integer<64, signed>):
// CHECK-NEXT:     %[[SUM:.*]] = polang.add %[[X_ARG]], %[[Y_ARG]] : !polang.integer<64, signed>, !polang.integer<64, signed> -> !polang.integer<64, signed>
// CHECK-NEXT:     polang.yield %[[SUM]] : !polang.integer<64, signed>
// CHECK-NEXT:   }
%2 = polang.let_expr ["x", "y"] -> !polang.integer<64, signed>
  binding {
    polang.yield.binding %0 : !polang.integer<64, signed>
  }
  binding {
    polang.yield.binding %1 : !polang.integer<64, signed>
  }
  body {
  ^bb0(%x_arg: !polang.integer<64, signed>, %y_arg: !polang.integer<64, signed>):
    %sum = polang.add %x_arg, %y_arg : !polang.integer<64, signed>, !polang.integer<64, signed> -> !polang.integer<64, signed>
    polang.yield %sum : !polang.integer<64, signed>
  }

// Test let expression with single binding
// CHECK:       %[[C5:.*]] = polang.constant.integer 5 : !polang.integer<64, signed>
%3 = polang.constant.integer 5 : !polang.integer<64, signed>
// CHECK-NEXT:   %[[SINGLE:.*]] = polang.let_expr ["n"] -> !polang.integer<64, signed>
// CHECK-NEXT:   binding {
// CHECK-NEXT:     polang.yield.binding %[[C5]] : !polang.integer<64, signed>
// CHECK-NEXT:   }
// CHECK-NEXT:   body {
// CHECK-NEXT:   ^bb0(%[[N_ARG:.*]]: !polang.integer<64, signed>):
// CHECK-NEXT:     polang.yield %[[N_ARG]] : !polang.integer<64, signed>
// CHECK-NEXT:   }
%4 = polang.let_expr ["n"] -> !polang.integer<64, signed>
  binding {
    polang.yield.binding %3 : !polang.integer<64, signed>
  }
  body {
  ^bb0(%n_arg: !polang.integer<64, signed>):
    polang.yield %n_arg : !polang.integer<64, signed>
  }

// CHECK-NEXT: }
