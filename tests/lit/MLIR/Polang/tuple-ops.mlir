// RUN: %polang_opt %s | %polang_opt | %FileCheck %s

// Test round-trip of tuple operations in Polang dialect

// CHECK: module {
module {
  // CHECK:   polang.func @test_tuple_ops() -> !polang.integer<64, signed> {
  polang.func @test_tuple_ops() -> !polang.integer<64, signed> {
    // CHECK:     %[[C1:.*]] = polang.constant.integer 42 : !polang.integer<64, signed>
    %0 = polang.constant.integer 42 : !polang.integer<64, signed>
    // CHECK:     %[[C2:.*]] = polang.constant.float 3.140000e+00 : !polang.float<64>
    %1 = polang.constant.float 3.14 : !polang.float<64>
    // CHECK:     %[[C3:.*]] = polang.constant.bool true : !polang.bool
    %2 = polang.constant.bool true : !polang.bool

    // CHECK:     %[[TUPLE:.*]] = polang.tuple %[[C1]], %[[C2]], %[[C3]] : (!polang.integer<64, signed>, !polang.float<64>, !polang.bool) -> !polang.tuple<!polang.integer<64, signed>, !polang.float<64>, !polang.bool>
    %3 = polang.tuple %0, %1, %2 : (!polang.integer<64, signed>, !polang.float<64>, !polang.bool) -> !polang.tuple<!polang.integer<64, signed>, !polang.float<64>, !polang.bool>

    // CHECK:     %[[ELEM0:.*]] = polang.tuple.get %[[TUPLE]][0] : <!polang.integer<64, signed>, !polang.float<64>, !polang.bool> -> !polang.integer<64, signed>
    %4 = polang.tuple.get %3[0] : <!polang.integer<64, signed>, !polang.float<64>, !polang.bool> -> !polang.integer<64, signed>

    // CHECK:     %[[EMPTY_TUPLE:.*]] = polang.tuple : () -> !polang.tuple<>
    %5 = polang.tuple : () -> !polang.tuple<>

    // CHECK:     polang.return %[[ELEM0]] : !polang.integer<64, signed>
    polang.return %4 : !polang.integer<64, signed>
  }

  // CHECK:   polang.generic_func @test_generic_tuple<a, b>(%arg0: !polang.type_param<"a">, %arg1: !polang.type_param<"b">) -> !polang.tuple<!polang.type_param<"a">, !polang.type_param<"b">> {
  polang.generic_func @test_generic_tuple<a, b>(%arg0: !polang.type_param<"a">, %arg1: !polang.type_param<"b">) -> !polang.tuple<!polang.type_param<"a">, !polang.type_param<"b">> {
    // CHECK:     %[[PAIR:.*]] = polang.tuple %arg0, %arg1 : (!polang.type_param<"a">, !polang.type_param<"b">) -> !polang.tuple<!polang.type_param<"a">, !polang.type_param<"b">>
    %0 = polang.tuple %arg0, %arg1 : (!polang.type_param<"a">, !polang.type_param<"b">) -> !polang.tuple<!polang.type_param<"a">, !polang.type_param<"b">>
    // CHECK:     polang.return %[[PAIR]] : !polang.tuple<!polang.type_param<"a">, !polang.type_param<"b">>
    polang.return %0 : !polang.tuple<!polang.type_param<"a">, !polang.type_param<"b">>
  }
}
