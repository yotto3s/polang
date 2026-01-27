// RUN: %polang_opt --allow-unregistered-dialect %s | %polang_opt --allow-unregistered-dialect | %FileCheck %s

// Test polang_ast.typevar type parsing and printing

// CHECK: module {

// Test type variable with default kind (any)
// CHECK-NEXT:   %0 = "test.test_type"() : () -> !polang_ast.typevar<1>
"test.test_type"() : () -> !polang_ast.typevar<1>

// Test type variable with explicit any kind
// CHECK-NEXT:   %1 = "test.test_type"() : () -> !polang_ast.typevar<2>
"test.test_type"() : () -> !polang_ast.typevar<2, any>

// Test type variable with integer kind
// CHECK-NEXT:   %2 = "test.test_type"() : () -> !polang_ast.typevar<3, integer>
"test.test_type"() : () -> !polang_ast.typevar<3, integer>

// Test type variable with float kind
// CHECK-NEXT:   %3 = "test.test_type"() : () -> !polang_ast.typevar<4, float>
"test.test_type"() : () -> !polang_ast.typevar<4, float>

// Test higher type variable IDs
// CHECK-NEXT:   %4 = "test.test_type"() : () -> !polang_ast.typevar<100>
"test.test_type"() : () -> !polang_ast.typevar<100>

// CHECK-NEXT:   %5 = "test.test_type"() : () -> !polang_ast.typevar<999, integer>
"test.test_type"() : () -> !polang_ast.typevar<999, integer>

// CHECK-NEXT: }
