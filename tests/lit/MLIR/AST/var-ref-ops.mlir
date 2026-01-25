// RUN: %polang_opt %s | %polang_opt | %FileCheck %s

// Test polang_ast variable reference operations

// CHECK: module {

// Test simple variable reference
// CHECK-NEXT:   %0 = polang_ast.var_ref "x" : !polang_ast.typevar<1>
%x = polang_ast.var_ref "x" : !polang_ast.typevar<1, any>

// Test variable reference with concrete type
// CHECK-NEXT:   %1 = polang_ast.var_ref "y" : !polang.integer<64, signed>
%y = polang_ast.var_ref "y" : !polang.integer<64, signed>

// Test qualified reference (module.member)
// CHECK-NEXT:   %2 = polang_ast.qualified_ref ["Math", "pi"] : !polang.float<64>
%math_pi = polang_ast.qualified_ref ["Math", "pi"] : !polang.float<64>

// Test longer qualified reference
// CHECK-NEXT:   %3 = polang_ast.qualified_ref ["std", "io", "print"] : !polang_ast.typevar<2>
%std_io_print = polang_ast.qualified_ref ["std", "io", "print"] : !polang_ast.typevar<2, any>

// CHECK-NEXT: }
