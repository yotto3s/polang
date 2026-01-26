// RUN: %polang_opt %s | %polang_opt | %FileCheck %s

// Test yield.binding operation (singular, for separate binding regions)

// CHECK: module {

%0 = polang_ast.constant.integer 42 : !polang.integer<64, signed>

// CHECK:         %[[C42:.*]] = polang_ast.constant.integer 42 : !polang.integer<64, signed>
// CHECK-NEXT:    %[[RESULT:.*]] = polang_ast.let_expr ["x"] -> !polang.integer<64, signed>
// CHECK-NEXT:    binding {
// CHECK-NEXT:      polang_ast.yield.binding %[[C42]] : !polang.integer<64, signed>
// CHECK-NEXT:    }
// CHECK-NEXT:    body {
// CHECK-NEXT:    ^bb0(%{{.*}}: !polang.integer<64, signed>):
// CHECK-NEXT:      polang_ast.yield %{{.*}} : !polang.integer<64, signed>
// CHECK-NEXT:    }
%1 = polang_ast.let_expr ["x"] -> !polang.integer<64, signed>
  binding {
    polang_ast.yield.binding %0 : !polang.integer<64, signed>
  }
  body {
  ^bb0(%x: !polang.integer<64, signed>):
    polang_ast.yield %x : !polang.integer<64, signed>
  }

// CHECK-NEXT:  }
