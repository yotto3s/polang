// RUN: %polang_opt --allow-unregistered-dialect %s | %polang_opt --allow-unregistered-dialect | %FileCheck %s

// Test polang_ast control flow operations

// CHECK: module {

// CHECK-NEXT:   %0 = polang_ast.constant.bool true : !polang.bool
%0 = polang_ast.constant.bool true : !polang.bool
// CHECK-NEXT:   %1 = polang_ast.constant.integer 10 : !polang.integer<64, signed>
%1 = polang_ast.constant.integer 10 : !polang.integer<64, signed>
// CHECK-NEXT:   %2 = polang_ast.constant.integer 20 : !polang.integer<64, signed>
%2 = polang_ast.constant.integer 20 : !polang.integer<64, signed>

// Test if operation with concrete types
// CHECK-NEXT:   %3 = polang_ast.if %0 : !polang.bool -> !polang.integer<64, signed> {
// CHECK-NEXT:     polang_ast.yield %1 : !polang.integer<64, signed>
// CHECK-NEXT:   } else {
// CHECK-NEXT:     polang_ast.yield %2 : !polang.integer<64, signed>
// CHECK-NEXT:   }
%3 = polang_ast.if %0 : !polang.bool -> !polang.integer<64, signed> {
  polang_ast.yield %1 : !polang.integer<64, signed>
} else {
  polang_ast.yield %2 : !polang.integer<64, signed>
}

// Test if with type variable result
// CHECK-NEXT:   %4 = polang_ast.constant.integer 5 : !polang_ast.typevar<1, integer>
%4 = polang_ast.constant.integer 5 : !polang_ast.typevar<1, integer>
// CHECK-NEXT:   %5 = polang_ast.constant.integer 3 : !polang_ast.typevar<1, integer>
%5 = polang_ast.constant.integer 3 : !polang_ast.typevar<1, integer>
// CHECK-NEXT:   %6 = polang_ast.if %0 : !polang.bool -> !polang_ast.typevar<1, integer> {
// CHECK-NEXT:     polang_ast.yield %4 : !polang_ast.typevar<1, integer>
// CHECK-NEXT:   } else {
// CHECK-NEXT:     polang_ast.yield %5 : !polang_ast.typevar<1, integer>
// CHECK-NEXT:   }
%6 = polang_ast.if %0 : !polang.bool -> !polang_ast.typevar<1, integer> {
  polang_ast.yield %4 : !polang_ast.typevar<1, integer>
} else {
  polang_ast.yield %5 : !polang_ast.typevar<1, integer>
}

// Test if with unconstrained type variable condition (any kind can unify with bool)
// CHECK-NEXT:   %7 = "test.unknown_bool"() : () -> !polang_ast.typevar<2>
%7 = "test.unknown_bool"() : () -> !polang_ast.typevar<2>
// CHECK-NEXT:   %8 = polang_ast.if %7 : !polang_ast.typevar<2> -> !polang.integer<64, signed> {
// CHECK-NEXT:     polang_ast.yield %1 : !polang.integer<64, signed>
// CHECK-NEXT:   } else {
// CHECK-NEXT:     polang_ast.yield %2 : !polang.integer<64, signed>
// CHECK-NEXT:   }
%8 = polang_ast.if %7 : !polang_ast.typevar<2> -> !polang.integer<64, signed> {
  polang_ast.yield %1 : !polang.integer<64, signed>
} else {
  polang_ast.yield %2 : !polang.integer<64, signed>
}

// CHECK-NEXT: }
