// RUN: %polang_opt --polang-monomorphize %s | %FileCheck %s

// Test monomorphization of generic functions taking and returning tuples

// CHECK-LABEL: polang.func @make_pair$i64_f64(%arg0: !polang.integer<64, signed>, %arg1: !polang.float<64>) -> !polang.tuple<!polang.integer<64, signed>, !polang.float<64>> {
// CHECK-NEXT:    %[[TUP:.*]] = polang.tuple %arg0, %arg1 : (!polang.integer<64, signed>, !polang.float<64>) -> !polang.tuple<!polang.integer<64, signed>, !polang.float<64>>
// CHECK-NEXT:    polang.return %[[TUP]] : !polang.tuple<!polang.integer<64, signed>, !polang.float<64>>
// CHECK-NEXT:  }

// CHECK-LABEL: polang.func @swap$tup2_i64_f64(%arg0: !polang.tuple<!polang.integer<64, signed>, !polang.float<64>>) -> !polang.tuple<!polang.float<64>, !polang.integer<64, signed>> {
// CHECK-NEXT:    %[[E0:.*]] = polang.tuple.get %arg0[0] : <!polang.integer<64, signed>, !polang.float<64>> -> !polang.integer<64, signed>
// CHECK-NEXT:    %[[E1:.*]] = polang.tuple.get %arg0[1] : <!polang.integer<64, signed>, !polang.float<64>> -> !polang.float<64>
// CHECK-NEXT:    %[[RES:.*]] = polang.tuple %[[E1]], %[[E0]] : (!polang.float<64>, !polang.integer<64, signed>) -> !polang.tuple<!polang.float<64>, !polang.integer<64, signed>>
// CHECK-NEXT:    polang.return %[[RES]] : !polang.tuple<!polang.float<64>, !polang.integer<64, signed>>
// CHECK-NEXT:  }

// CHECK-LABEL: polang.func @__polang_entry() -> !polang.integer<64, signed> {
// CHECK:         %[[PAIR:.*]] = polang.call @make_pair$i64_f64(%{{.*}}, %{{.*}}) : (!polang.integer<64, signed>, !polang.float<64>) -> !polang.tuple<!polang.integer<64, signed>, !polang.float<64>>
// CHECK:         %[[SWAPPED:.*]] = polang.call @swap$tup2_i64_f64(%[[PAIR]]) : (!polang.tuple<!polang.integer<64, signed>, !polang.float<64>>) -> !polang.tuple<!polang.float<64>, !polang.integer<64, signed>>
// CHECK:         %[[FIRST:.*]] = polang.tuple.get %[[SWAPPED]][1] : <!polang.float<64>, !polang.integer<64, signed>> -> !polang.integer<64, signed>
// CHECK:         polang.return %[[FIRST]] : !polang.integer<64, signed>
// CHECK:       }

module {
  polang.generic_func @make_pair<a, b>(%x: !polang.type_param<"a">, %y: !polang.type_param<"b">) -> !polang.tuple<!polang.type_param<"a">, !polang.type_param<"b">> {
    %0 = polang.tuple %x, %y : (!polang.type_param<"a">, !polang.type_param<"b">) -> !polang.tuple<!polang.type_param<"a">, !polang.type_param<"b">>
    polang.return %0 : !polang.tuple<!polang.type_param<"a">, !polang.type_param<"b">>
  }

  polang.generic_func @swap<a, b>(%pair: !polang.tuple<!polang.type_param<"a">, !polang.type_param<"b">>) -> !polang.tuple<!polang.type_param<"b">, !polang.type_param<"a">> {
    %first = polang.tuple.get %pair[0] : !polang.tuple<!polang.type_param<"a">, !polang.type_param<"b">> -> !polang.type_param<"a">
    %second = polang.tuple.get %pair[1] : !polang.tuple<!polang.type_param<"a">, !polang.type_param<"b">> -> !polang.type_param<"b">
    %res = polang.tuple %second, %first : (!polang.type_param<"b">, !polang.type_param<"a">) -> !polang.tuple<!polang.type_param<"b">, !polang.type_param<"a">>
    polang.return %res : !polang.tuple<!polang.type_param<"b">, !polang.type_param<"a">>
  }

  polang.func @__polang_entry() -> !polang.integer<64, signed> {
    %c1 = polang.constant.integer 42 : !polang.integer<64, signed>
    %c2 = polang.constant.float 3.14 : !polang.float<64>
    %p = polang.instantiate @make_pair<a = !polang.integer<64, signed>, b = !polang.float<64>>(%c1, %c2) : (!polang.integer<64, signed>, !polang.float<64>) -> !polang.tuple<!polang.integer<64, signed>, !polang.float<64>>
    %s = polang.instantiate @swap<a = !polang.integer<64, signed>, b = !polang.float<64>>(%p) : (!polang.tuple<!polang.integer<64, signed>, !polang.float<64>>) -> !polang.tuple<!polang.float<64>, !polang.integer<64, signed>>
    %ret = polang.tuple.get %s[1] : !polang.tuple<!polang.float<64>, !polang.integer<64, signed>> -> !polang.integer<64, signed>
    polang.return %ret : !polang.integer<64, signed>
  }
}
