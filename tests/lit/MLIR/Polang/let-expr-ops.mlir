// RUN: %polang_opt %s | %polang_opt | %FileCheck %s

// Test polang let expression operations with variadic binding regions

// CHECK: module {

// Test let expression with two bindings
// CHECK-NEXT:   %[[C10:.*]] = polang.constant.integer 10 : si64
%0 = polang.constant.integer 10 : si64
// CHECK-NEXT:   %[[C20:.*]] = polang.constant.integer 20 : si64
%1 = polang.constant.integer 20 : si64
// CHECK-NEXT:   %[[RESULT:.*]] = polang.let_expr ["x", "y"] -> si64
// CHECK-NEXT:   binding {
// CHECK-NEXT:     polang.yield.binding %[[C10]] : si64
// CHECK-NEXT:   }
// CHECK-NEXT:   binding {
// CHECK-NEXT:     polang.yield.binding %[[C20]] : si64
// CHECK-NEXT:   }
// CHECK-NEXT:   body {
// CHECK-NEXT:   ^bb0(%[[X_ARG:.*]]: si64, %[[Y_ARG:.*]]: si64):
// CHECK-NEXT:     %[[SUM:.*]] = polang.add %[[X_ARG]], %[[Y_ARG]] : si64, si64 -> si64
// CHECK-NEXT:     polang.yield %[[SUM]] : si64
// CHECK-NEXT:   }
%2 = polang.let_expr ["x", "y"] -> si64
  binding {
    polang.yield.binding %0 : si64
  }
  binding {
    polang.yield.binding %1 : si64
  }
  body {
  ^bb0(%x_arg: si64, %y_arg: si64):
    %sum = polang.add %x_arg, %y_arg : si64, si64 -> si64
    polang.yield %sum : si64
  }

// Test let expression with single binding
// CHECK:       %[[C5:.*]] = polang.constant.integer 5 : si64
%3 = polang.constant.integer 5 : si64
// CHECK-NEXT:   %[[SINGLE:.*]] = polang.let_expr ["n"] -> si64
// CHECK-NEXT:   binding {
// CHECK-NEXT:     polang.yield.binding %[[C5]] : si64
// CHECK-NEXT:   }
// CHECK-NEXT:   body {
// CHECK-NEXT:   ^bb0(%[[N_ARG:.*]]: si64):
// CHECK-NEXT:     polang.yield %[[N_ARG]] : si64
// CHECK-NEXT:   }
%4 = polang.let_expr ["n"] -> si64
  binding {
    polang.yield.binding %3 : si64
  }
  body {
  ^bb0(%n_arg: si64):
    polang.yield %n_arg : si64
  }

// CHECK-NEXT: }
