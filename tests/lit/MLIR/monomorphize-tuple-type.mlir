// RUN: %polang_opt --polang-monomorphize %s | %FileCheck %s

// Test monomorphization of generic functions taking and returning tuples

// CHECK: module {
// CHECK:   polang.func @identity$tup2_i64_f64(%arg0: !polang.tuple<!polang.integer<64, signed>, !polang.float<64>>) -> !polang.tuple<!polang.integer<64, signed>, !polang.float<64>> {
// CHECK:     polang.return %arg0 : !polang.tuple<!polang.integer<64, signed>, !polang.float<64>>
// CHECK:   }
// CHECK:   polang.func @nested_id$tup2_i64_tup2_f64_bool(%arg0: !polang.tuple<!polang.integer<64, signed>, !polang.tuple<!polang.float<64>, !polang.bool>>) -> !polang.tuple<!polang.integer<64, signed>, !polang.tuple<!polang.float<64>, !polang.bool>> {
// CHECK:     polang.return %arg0 : !polang.tuple<!polang.integer<64, signed>, !polang.tuple<!polang.float<64>, !polang.bool>>
// CHECK:   }
// CHECK:   polang.func @test_call(%arg0: !polang.tuple<!polang.integer<64, signed>, !polang.float<64>>, %arg1: !polang.tuple<!polang.integer<64, signed>, !polang.tuple<!polang.float<64>, !polang.bool>>) -> !polang.tuple<!polang.integer<64, signed>, !polang.tuple<!polang.float<64>, !polang.bool>> {
// CHECK:     %[[RES1:.*]] = polang.call @identity$tup2_i64_f64(%arg0) : (!polang.tuple<!polang.integer<64, signed>, !polang.float<64>>) -> !polang.tuple<!polang.integer<64, signed>, !polang.float<64>>
// CHECK:     %[[RES2:.*]] = polang.call @nested_id$tup2_i64_tup2_f64_bool(%arg1) : (!polang.tuple<!polang.integer<64, signed>, !polang.tuple<!polang.float<64>, !polang.bool>>) -> !polang.tuple<!polang.integer<64, signed>, !polang.tuple<!polang.float<64>, !polang.bool>>
// CHECK:     polang.return %[[RES2]] : !polang.tuple<!polang.integer<64, signed>, !polang.tuple<!polang.float<64>, !polang.bool>>
// CHECK:   }
// CHECK: }

module {
  polang.generic_func @identity<a, b>(%x: !polang.tuple<!polang.type_param<"a">, !polang.type_param<"b">>) -> !polang.tuple<!polang.type_param<"a">, !polang.type_param<"b">> {
    polang.return %x : !polang.tuple<!polang.type_param<"a">, !polang.type_param<"b">>
  }

  polang.generic_func @nested_id<a, b, c>(%x: !polang.tuple<!polang.type_param<"a">, !polang.tuple<!polang.type_param<"b">, !polang.type_param<"c">>>) -> !polang.tuple<!polang.type_param<"a">, !polang.tuple<!polang.type_param<"b">, !polang.type_param<"c">>> {
    polang.return %x : !polang.tuple<!polang.type_param<"a">, !polang.tuple<!polang.type_param<"b">, !polang.type_param<"c">>>
  }

  polang.func @test_call(%arg0: !polang.tuple<!polang.integer<64, signed>, !polang.float<64>>,
                         %arg1: !polang.tuple<!polang.integer<64, signed>, !polang.tuple<!polang.float<64>, !polang.bool>>)
      -> !polang.tuple<!polang.integer<64, signed>, !polang.tuple<!polang.float<64>, !polang.bool>> {
    %res1 = polang.instantiate @identity<a = !polang.integer<64, signed>, b = !polang.float<64>>(%arg0) : (!polang.tuple<!polang.integer<64, signed>, !polang.float<64>>) -> !polang.tuple<!polang.integer<64, signed>, !polang.float<64>>
    %res2 = polang.instantiate @nested_id<a = !polang.integer<64, signed>, b = !polang.float<64>, c = !polang.bool>(%arg1) : (!polang.tuple<!polang.integer<64, signed>, !polang.tuple<!polang.float<64>, !polang.bool>>) -> !polang.tuple<!polang.integer<64, signed>, !polang.tuple<!polang.float<64>, !polang.bool>>
    polang.return %res2 : !polang.tuple<!polang.integer<64, signed>, !polang.tuple<!polang.float<64>, !polang.bool>>
  }
}
