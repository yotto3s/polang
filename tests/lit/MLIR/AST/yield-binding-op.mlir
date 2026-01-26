// RUN: %polang_opt %s | %polang_opt | FileCheck %s

// Test yield.binding operation (singular, for separate binding regions)

%0 = polang_ast.constant.integer 42 : !polang.integer<64, signed>

// CHECK: polang_ast.let_expr ["x"] -> !polang.integer<64, signed>
// CHECK-NEXT: binding {
// CHECK-NEXT:   polang_ast.yield.binding %{{.*}} : !polang.integer<64, signed>
// CHECK-NEXT: }
// CHECK-NEXT: body {
%1 = polang_ast.let_expr ["x"] -> !polang.integer<64, signed>
  binding {
    polang_ast.yield.binding %0 : !polang.integer<64, signed>
  }
  body {
  ^bb0(%x: !polang.integer<64, signed>):
    polang_ast.yield %x : !polang.integer<64, signed>
  }
