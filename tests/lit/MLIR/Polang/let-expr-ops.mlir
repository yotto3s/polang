// RUN: %polang_opt %s | %polang_opt | %FileCheck %s

// Test polang let expression with separate binding regions

// CHECK: module {

%0 = polang.constant.integer 10 : !polang.integer<64, signed>
%1 = polang.constant.integer 20 : !polang.integer<64, signed>

// CHECK: polang.let_expr ["x", "y"] -> !polang.integer<64, signed>
// CHECK-NEXT: binding {
// CHECK-NEXT:   polang.yield.binding
// CHECK: }
// CHECK-NEXT: binding {
// CHECK-NEXT:   polang.yield.binding
// CHECK: }
// CHECK-NEXT: body {
// CHECK-NEXT: ^bb0(%{{.*}}: !polang.integer<64, signed>, %{{.*}}: !polang.integer<64, signed>):
// CHECK:   polang.yield
// CHECK-NEXT: }
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

// Single binding
%3 = polang.constant.integer 5 : !polang.integer<64, signed>

// CHECK: polang.let_expr ["n"] -> !polang.integer<64, signed>
%4 = polang.let_expr ["n"] -> !polang.integer<64, signed>
  binding {
    polang.yield.binding %3 : !polang.integer<64, signed>
  }
  body {
  ^bb0(%n_arg: !polang.integer<64, signed>):
    polang.yield %n_arg : !polang.integer<64, signed>
  }

// CHECK: }
