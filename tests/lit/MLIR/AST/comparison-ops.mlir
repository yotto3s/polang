// RUN: %polang_opt %s | %polang_opt | %FileCheck %s

// Test polang_ast comparison and cast operations

// CHECK: module {

// Test comparison with concrete types
// CHECK-NEXT:   %0 = polang_ast.constant.integer 10 : !polang.integer<64, signed>
%0 = polang_ast.constant.integer 10 : !polang.integer<64, signed>
// CHECK-NEXT:   %1 = polang_ast.constant.integer 20 : !polang.integer<64, signed>
%1 = polang_ast.constant.integer 20 : !polang.integer<64, signed>

// CHECK-NEXT:   %2 = polang_ast.cmp eq, %0, %1 : !polang.integer<64, signed>, !polang.integer<64, signed>
%2 = polang_ast.cmp eq, %0, %1 : !polang.integer<64, signed>, !polang.integer<64, signed>
// CHECK-NEXT:   %3 = polang_ast.cmp ne, %0, %1 : !polang.integer<64, signed>, !polang.integer<64, signed>
%3 = polang_ast.cmp ne, %0, %1 : !polang.integer<64, signed>, !polang.integer<64, signed>
// CHECK-NEXT:   %4 = polang_ast.cmp lt, %0, %1 : !polang.integer<64, signed>, !polang.integer<64, signed>
%4 = polang_ast.cmp lt, %0, %1 : !polang.integer<64, signed>, !polang.integer<64, signed>
// CHECK-NEXT:   %5 = polang_ast.cmp le, %0, %1 : !polang.integer<64, signed>, !polang.integer<64, signed>
%5 = polang_ast.cmp le, %0, %1 : !polang.integer<64, signed>, !polang.integer<64, signed>
// CHECK-NEXT:   %6 = polang_ast.cmp gt, %0, %1 : !polang.integer<64, signed>, !polang.integer<64, signed>
%6 = polang_ast.cmp gt, %0, %1 : !polang.integer<64, signed>, !polang.integer<64, signed>
// CHECK-NEXT:   %7 = polang_ast.cmp ge, %0, %1 : !polang.integer<64, signed>, !polang.integer<64, signed>
%7 = polang_ast.cmp ge, %0, %1 : !polang.integer<64, signed>, !polang.integer<64, signed>

// Test comparison with type variables
// CHECK-NEXT:   %8 = polang_ast.constant.integer 5 : !polang_ast.typevar<1, integer>
%8 = polang_ast.constant.integer 5 : !polang_ast.typevar<1, integer>
// CHECK-NEXT:   %9 = polang_ast.constant.integer 3 : !polang_ast.typevar<1, integer>
%9 = polang_ast.constant.integer 3 : !polang_ast.typevar<1, integer>
// CHECK-NEXT:   %10 = polang_ast.cmp lt, %8, %9 : !polang_ast.typevar<1, integer>, !polang_ast.typevar<1, integer>
%10 = polang_ast.cmp lt, %8, %9 : !polang_ast.typevar<1, integer>, !polang_ast.typevar<1, integer>

// Test cast operation
// CHECK-NEXT:   %11 = polang_ast.cast %0 : !polang.integer<64, signed> -> !polang.float<64>
%11 = polang_ast.cast %0 : !polang.integer<64, signed> -> !polang.float<64>

// Test cast with type variable
// CHECK-NEXT:   %12 = polang_ast.cast %8 : !polang_ast.typevar<1, integer> -> !polang.float<64>
%12 = polang_ast.cast %8 : !polang_ast.typevar<1, integer> -> !polang.float<64>

// CHECK-NEXT: }
