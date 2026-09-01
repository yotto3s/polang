// RUN: %polang_opt --convert-polang-to-standard %s | %FileCheck %s

// Test lowering of tuple operations, discarding of unit tuples, and sret convention

// CHECK: module {
module {
  // CHECK:   func.func @test_unit_func() {
  // CHECK:     return
  // CHECK:   }
  polang.func @test_unit_func(%arg: !polang.tuple<>) -> !polang.tuple<> {
    %unit = polang.tuple : () -> !polang.tuple<>
    polang.return %unit : !polang.tuple<>
  }

  // CHECK:   func.func @make_pair(%arg0: memref<2xi64>, %arg1: i64, %arg2: f64) {
  // CHECK:     %[[ALLOCA:.*]] = memref.alloca() : memref<2xi64>
  // CHECK:     %[[C0:.*]] = arith.constant 0 : index
  // CHECK:     memref.store %arg1, %[[ALLOCA]][%[[C0]]] : memref<2xi64>
  // CHECK:     %[[CAST_F64:.*]] = arith.bitcast %arg2 : f64 to i64
  // CHECK:     %[[C1:.*]] = arith.constant 1 : index
  // CHECK:     memref.store %[[CAST_F64]], %[[ALLOCA]][%[[C1]]] : memref<2xi64>
  // CHECK:     %[[IDX0:.*]] = arith.constant 0 : index
  // CHECK:     %[[V0:.*]] = memref.load %[[ALLOCA]][%[[IDX0]]] : memref<2xi64>
  // CHECK:     memref.store %[[V0]], %arg0[%[[IDX0]]] : memref<2xi64>
  // CHECK:     %[[IDX1:.*]] = arith.constant 1 : index
  // CHECK:     %[[V1:.*]] = memref.load %[[ALLOCA]][%[[IDX1]]] : memref<2xi64>
  // CHECK:     memref.store %[[V1]], %arg0[%[[IDX1]]] : memref<2xi64>
  // CHECK:     return
  // CHECK:   }
  polang.func @make_pair(%a: !polang.integer<64, signed>, %b: !polang.float<64>) -> !polang.tuple<!polang.integer<64, signed>, !polang.float<64>> {
    %t = polang.tuple %a, %b : (!polang.integer<64, signed>, !polang.float<64>) -> !polang.tuple<!polang.integer<64, signed>, !polang.float<64>>
    polang.return %t : !polang.tuple<!polang.integer<64, signed>, !polang.float<64>>
  }

  // CHECK:   func.func @caller(%arg0: i64, %arg1: f64) -> i64 {
  // CHECK:     %[[CALLER_BUF:.*]] = memref.alloca() : memref<2xi64>
  // CHECK:     call @make_pair(%[[CALLER_BUF]], %arg0, %arg1) : (memref<2xi64>, i64, f64) -> ()
  // CHECK:     %[[LOAD_C0:.*]] = arith.constant 0 : index
  // CHECK:     %[[LOAD_VAL:.*]] = memref.load %[[CALLER_BUF]][%[[LOAD_C0]]] : memref<2xi64>
  // CHECK:     return %[[LOAD_VAL]] : i64
  // CHECK:   }
  polang.func @caller(%a: !polang.integer<64, signed>, %b: !polang.float<64>) -> !polang.integer<64, signed> {
    %pair = polang.call @make_pair(%a, %b) : (!polang.integer<64, signed>, !polang.float<64>) -> !polang.tuple<!polang.integer<64, signed>, !polang.float<64>>
    %first = polang.tuple.get %pair[0] : !polang.tuple<!polang.integer<64, signed>, !polang.float<64>> -> !polang.integer<64, signed>
    polang.return %first : !polang.integer<64, signed>
  }
}
