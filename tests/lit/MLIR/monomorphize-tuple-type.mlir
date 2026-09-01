// RUN: %polang_opt --polang-monomorphize %s | %FileCheck %s

// Test monomorphization of generic functions taking and returning tuples

// CHECK: module {
// CHECK:   polang.func @identity$tup2_i64_f64(%arg0: !polang.tuple<!polang.integer<64, signed>, !polang.float<64>>) -> !polang.tuple<!polang.integer<64, signed>, !polang.float<64>> {
// CHECK:     polang.return %arg0 : !polang.tuple<!polang.integer<64, signed>, !polang.float<64>>
// CHECK:   }
// CHECK:   polang.func @test_call(%arg0: !polang.tuple<!polang.integer<64, signed>, !polang.float<64>>) -> !polang.tuple<!polang.integer<64, signed>, !polang.float<64>> {
// CHECK:     %[[RES:.*]] = polang.call @identity$tup2_i64_f64(%arg0) : (!polang.tuple<!polang.integer<64, signed>, !polang.float<64>>) -> !polang.tuple<!polang.integer<64, signed>, !polang.float<64>>
// CHECK:     polang.return %[[RES]] : !polang.tuple<!polang.integer<64, signed>, !polang.float<64>>
// CHECK:   }
// CHECK: }

module {
  polang.generic_func @identity<a, b>(%x: !polang.tuple<!polang.type_param<"a">, !polang.type_param<"b">>) -> !polang.tuple<!polang.type_param<"a">, !polang.type_param<"b">> {
    polang.return %x : !polang.tuple<!polang.type_param<"a">, !polang.type_param<"b">>
  }

  polang.func @test_call(%arg: !polang.tuple<!polang.integer<64, signed>, !polang.float<64>>) -> !polang.tuple<!polang.integer<64, signed>, !polang.float<64>> {
    %res = polang.instantiate @identity<a = !polang.integer<64, signed>, b = !polang.float<64>>(%arg) : (!polang.tuple<!polang.integer<64, signed>, !polang.float<64>>) -> !polang.tuple<!polang.integer<64, signed>, !polang.float<64>>
    polang.return %res : !polang.tuple<!polang.integer<64, signed>, !polang.float<64>>
  }
}
