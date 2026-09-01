// RUN: %polang_opt --polang-monomorphize %s | %FileCheck %s

// Test monomorphization of generic functions taking and returning tuples

// CHECK: module {
// CHECK:   polang.func @identity$tup2_i64_f64(%arg0: tuple<si64, f64>) -> tuple<si64, f64> {
// CHECK:     polang.return %arg0 : tuple<si64, f64>
// CHECK:   }
// CHECK:   polang.func @nested_id$tup2_i64_tup2_f64_bool(%arg0: tuple<si64, tuple<f64, i1>>) -> tuple<si64, tuple<f64, i1>> {
// CHECK:     polang.return %arg0 : tuple<si64, tuple<f64, i1>>
// CHECK:   }
// CHECK:   polang.func @test_call(%arg0: tuple<si64, f64>, %arg1: tuple<si64, tuple<f64, i1>>) -> tuple<si64, tuple<f64, i1>> {
// CHECK:     %[[RES1:.*]] = polang.call @identity$tup2_i64_f64(%arg0) : (tuple<si64, f64>) -> tuple<si64, f64>
// CHECK:     %[[RES2:.*]] = polang.call @nested_id$tup2_i64_tup2_f64_bool(%arg1) : (tuple<si64, tuple<f64, i1>>) -> tuple<si64, tuple<f64, i1>>
// CHECK:     polang.return %[[RES2]] : tuple<si64, tuple<f64, i1>>
// CHECK:   }
// CHECK: }

module {
  polang.generic_func @identity<a, b>(%x: tuple<!polang.type_param<"a">, !polang.type_param<"b">>) -> tuple<!polang.type_param<"a">, !polang.type_param<"b">> {
    polang.return %x : tuple<!polang.type_param<"a">, !polang.type_param<"b">>
  }

  polang.generic_func @nested_id<a, b, c>(%x: tuple<!polang.type_param<"a">, tuple<!polang.type_param<"b">, !polang.type_param<"c">>>) -> tuple<!polang.type_param<"a">, tuple<!polang.type_param<"b">, !polang.type_param<"c">>> {
    polang.return %x : tuple<!polang.type_param<"a">, tuple<!polang.type_param<"b">, !polang.type_param<"c">>>
  }

  polang.func @test_call(%arg0: tuple<si64, f64>,
                         %arg1: tuple<si64, tuple<f64, i1>>)
      -> tuple<si64, tuple<f64, i1>> {
    %res1 = polang.instantiate @identity<a = si64, b = f64>(%arg0) : (tuple<si64, f64>) -> tuple<si64, f64>
    %res2 = polang.instantiate @nested_id<a = si64, b = f64, c = i1>(%arg1) : (tuple<si64, tuple<f64, i1>>) -> tuple<si64, tuple<f64, i1>>
    polang.return %res2 : tuple<si64, tuple<f64, i1>>
  }
}
