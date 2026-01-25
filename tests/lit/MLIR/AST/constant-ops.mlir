// RUN: %polang_opt %s | %polang_opt | %FileCheck %s

// Test polang_ast constant operations

// CHECK: module {

// Test integer constants with concrete type
// CHECK-NEXT:   %0 = polang_ast.constant.integer 42 : !polang.integer<64, signed>
%0 = polang_ast.constant.integer 42 : !polang.integer<64, signed>

// CHECK-NEXT:   %1 = polang_ast.constant.integer 255 : !polang.integer<8, unsigned>
%1 = polang_ast.constant.integer 255 : !polang.integer<8, unsigned>

// Test integer constant with type variable
// CHECK-NEXT:   %2 = polang_ast.constant.integer 100 : !polang_ast.typevar<1, integer>
%2 = polang_ast.constant.integer 100 : !polang_ast.typevar<1, integer>

// Test float constants with concrete type
// CHECK-NEXT:   %3 = polang_ast.constant.float 3.140000e+00 : !polang.float<64>
%3 = polang_ast.constant.float 3.14 : !polang.float<64>

// CHECK-NEXT:   %4 = polang_ast.constant.float 1.500000e+00 : !polang.float<32>
%4 = polang_ast.constant.float 1.5 : !polang.float<32>

// Test float constant with type variable
// CHECK-NEXT:   %5 = polang_ast.constant.float 2.718000e+00 : !polang_ast.typevar<2, float>
%5 = polang_ast.constant.float 2.718 : !polang_ast.typevar<2, float>

// Test boolean constants
// CHECK-NEXT:   %6 = polang_ast.constant.bool true : !polang.bool
%6 = polang_ast.constant.bool true : !polang.bool

// CHECK-NEXT:   %7 = polang_ast.constant.bool false : !polang.bool
%7 = polang_ast.constant.bool false : !polang.bool

// CHECK-NEXT: }
