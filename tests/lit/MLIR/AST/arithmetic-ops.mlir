// RUN: %polang_opt %s | %polang_opt | %FileCheck %s

// Test polang_ast arithmetic operations

// CHECK: module {

// Test add operation with concrete types
// CHECK-NEXT:   %0 = polang_ast.constant.integer 10 : !polang.integer<64, signed>
%0 = polang_ast.constant.integer 10 : !polang.integer<64, signed>
// CHECK-NEXT:   %1 = polang_ast.constant.integer 20 : !polang.integer<64, signed>
%1 = polang_ast.constant.integer 20 : !polang.integer<64, signed>
// CHECK-NEXT:   %2 = polang_ast.add %0, %1 : !polang.integer<64, signed>, !polang.integer<64, signed> -> !polang.integer<64, signed>
%2 = polang_ast.add %0, %1 : !polang.integer<64, signed>, !polang.integer<64, signed> -> !polang.integer<64, signed>

// Test sub operation
// CHECK-NEXT:   %3 = polang_ast.sub %0, %1 : !polang.integer<64, signed>, !polang.integer<64, signed> -> !polang.integer<64, signed>
%3 = polang_ast.sub %0, %1 : !polang.integer<64, signed>, !polang.integer<64, signed> -> !polang.integer<64, signed>

// Test mul operation
// CHECK-NEXT:   %4 = polang_ast.mul %0, %1 : !polang.integer<64, signed>, !polang.integer<64, signed> -> !polang.integer<64, signed>
%4 = polang_ast.mul %0, %1 : !polang.integer<64, signed>, !polang.integer<64, signed> -> !polang.integer<64, signed>

// Test div operation
// CHECK-NEXT:   %5 = polang_ast.div %0, %1 : !polang.integer<64, signed>, !polang.integer<64, signed> -> !polang.integer<64, signed>
%5 = polang_ast.div %0, %1 : !polang.integer<64, signed>, !polang.integer<64, signed> -> !polang.integer<64, signed>

// Test with float types
// CHECK-NEXT:   %6 = polang_ast.constant.float 1.500000e+00 : !polang.float<64>
%6 = polang_ast.constant.float 1.5 : !polang.float<64>
// CHECK-NEXT:   %7 = polang_ast.constant.float 2.500000e+00 : !polang.float<64>
%7 = polang_ast.constant.float 2.5 : !polang.float<64>
// CHECK-NEXT:   %8 = polang_ast.add %6, %7 : !polang.float<64>, !polang.float<64> -> !polang.float<64>
%8 = polang_ast.add %6, %7 : !polang.float<64>, !polang.float<64> -> !polang.float<64>

// Test with type variables (before type inference)
// CHECK-NEXT:   %9 = polang_ast.constant.integer 5 : !polang_ast.typevar<1, integer>
%9 = polang_ast.constant.integer 5 : !polang_ast.typevar<1, integer>
// CHECK-NEXT:   %10 = polang_ast.constant.integer 3 : !polang_ast.typevar<1, integer>
%10 = polang_ast.constant.integer 3 : !polang_ast.typevar<1, integer>
// CHECK-NEXT:   %11 = polang_ast.add %9, %10 : !polang_ast.typevar<1, integer>, !polang_ast.typevar<1, integer> -> !polang_ast.typevar<1, integer>
%11 = polang_ast.add %9, %10 : !polang_ast.typevar<1, integer>, !polang_ast.typevar<1, integer> -> !polang_ast.typevar<1, integer>

// CHECK-NEXT: }
