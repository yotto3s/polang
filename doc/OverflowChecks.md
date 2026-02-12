# Integer Overflow Checks

## Overview

Polang implements Swift-style overflow checking for all integer arithmetic operations. Unlike many languages that either silently wrap (C/C++) or only check in debug mode, Polang checks for overflow in all build modes and immediately terminates with an error message.

## Checked Operations

The following operations are checked for overflow:
- Addition (`+`)
- Subtraction (`-`)
- Multiplication (`*`)
- Unary negation (`-x`)
- Division (`MIN_INT / -1` for signed types)

## Implementation

### Architecture

The overflow checking is implemented as a late-stage MLIR transformation pass that runs after all optimizations but before final LLVM lowering:

1. **Early stages**: Use standard `arith.addi/subi/muli` operations
2. **MLIR optimizations**: All standard optimizations run on these operations (affine analysis, constant folding, CSE, canonicalization)
3. **InsertOverflowChecks pass**: Replaces arithmetic operations with LLVM overflow intrinsics
4. **LLVM lowering**: Final conversion to LLVM IR

### Signedness Tracking

Since MLIR's `arith` dialect doesn't distinguish between signed and unsigned for addition/subtraction/multiplication, we attach a `polang.is_unsigned` attribute during the PolangToStandard lowering pass. This attribute is read by the InsertOverflowChecks pass to choose the correct intrinsic:

- Signed: `llvm.sadd.with.overflow.iN`, `llvm.ssub.with.overflow.iN`, `llvm.smul.with.overflow.iN`
- Unsigned: `llvm.uadd.with.overflow.iN`, `llvm.usub.with.overflow.iN`, `llvm.umul.with.overflow.iN`

### Runtime Behavior

When overflow is detected:
1. The overflow intrinsic's flag is set
2. Control branches to an error block
3. The error block calls `__polang_runtime_error(message, line, column)`
4. The runtime function prints an error message and exits with status 1

Example error output:
```
Runtime error: integer overflow at line 5, column 10
```

## Performance

While overflow checking adds runtime overhead, the late-stage transformation approach ensures:
- MLIR optimizations can eliminate redundant checks
- Constant expressions are folded at compile time (no runtime check)
- LLVM can optimize the control flow
- The intrinsics are well-optimized by LLVM

For performance-critical code that intentionally uses wrapping behavior, future versions will support explicit wrapping operators (`&+`, `&-`, `&*`).

## Testing

Tests are located in `tests/lit/Execution/overflow-*.po` and verify:
- Signed overflow detection
- Unsigned overflow detection  
- All arithmetic operations
- Non-overflowing operations still work correctly

## References

- Spec: `spec/polang-spec.md` §6.1 (Integer Types)
- Issue: yotto3s/polang#XX (Add runtime check for integer overflow)
- Pass implementation: `mlir/lib/Transforms/InsertOverflowChecks.cpp`
- Runtime handler: `runtime/src/runtime.c`
