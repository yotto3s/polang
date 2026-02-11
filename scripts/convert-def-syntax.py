#!/usr/bin/env python3
"""Convert Polang files from def-based syntax to Haskell-style definitions.

Transforms:
  def x = expr                          -> x = expr
  def x: TYPE = expr                    -> x : TYPE\n x = expr
  def f(a: T1, b: T2): RET = expr      -> f : T1 * T2 -> RET\n f(a, b) = expr
  def f(a: T1): RET = expr             -> f : T1 -> RET\n f(a) = expr
  def f(a, b) = expr                   -> f(a, b) = expr
  def f(a) = expr                      -> f(a) = expr
  def f(): RET = expr                  -> f : RET\n f() = expr

Usage:
  python3 convert-def-syntax.py FILE           # print to stdout
  python3 convert-def-syntax.py FILE --inplace # modify in place
"""
import re
import sys


def parse_params(param_str):
    """Parse parameter string, handling nested parens.
    Returns list of (name, type_or_None) tuples."""
    params = []
    depth = 0
    current = ""
    for ch in param_str:
        if ch == "(":
            depth += 1
            current += ch
        elif ch == ")":
            depth -= 1
            current += ch
        elif ch == "," and depth == 0:
            params.append(current.strip())
            current = ""
        else:
            current += ch
    if current.strip():
        params.append(current.strip())

    result = []
    for p in params:
        if ":" in p:
            name, typ = p.split(":", 1)
            result.append((name.strip(), typ.strip()))
        else:
            result.append((p.strip(), None))
    return result


def convert_def_line(line, continuation_lines=None):
    """Convert a single def statement. Returns list of replacement lines."""
    if continuation_lines is None:
        continuation_lines = []

    stripped = line.lstrip()
    indent = line[: len(line) - len(stripped)]

    # Must start with 'def '
    if not stripped.startswith("def "):
        return [line] + continuation_lines

    rest = stripped[4:]  # after 'def '

    # Find inline comment (not inside parens or strings)
    def split_comment(s):
        """Split off trailing (* ... *) comment."""
        # Find the last (* that has a matching *)
        idx = s.rfind("(*")
        if idx >= 0 and s.rstrip().endswith("*)"):
            return s[:idx].rstrip(), s[idx:]
        return s, ""

    # Pattern 1: def name(params): RET = body
    # Pattern 2: def name(params) = body  (no types)
    # Pattern 3: def name: TYPE = body
    # Pattern 4: def name = body

    # Try function definition: name(...)
    m = re.match(r"(\w+)\(", rest)
    if m:
        func_name = m.group(1)
        # Find matching close paren
        depth = 0
        i = len(func_name)
        paren_start = i
        for i in range(paren_start, len(rest)):
            if rest[i] == "(":
                depth += 1
            elif rest[i] == ")":
                depth -= 1
                if depth == 0:
                    break
        param_str = rest[paren_start + 1 : i]
        after_paren = rest[i + 1 :]

        params = parse_params(param_str)
        has_types = any(t is not None for t in [p[1] for p in params])

        # Check for return type: ): RET = body
        after_paren_stripped = after_paren.lstrip()
        ret_type = None
        if after_paren_stripped.startswith(":"):
            # Has return type annotation
            after_colon = after_paren_stripped[1:].lstrip()
            # Find the = sign (could be at this line or continuation)
            eq_idx = after_colon.find("=")
            if eq_idx >= 0:
                ret_type = after_colon[:eq_idx].rstrip()
                body_with_comment = after_colon[eq_idx + 1 :].lstrip()
            else:
                # = might be on continuation line or at end of type
                ret_type = after_colon.rstrip()
                body_with_comment = ""
        elif after_paren_stripped.startswith("="):
            body_with_comment = after_paren_stripped[1:].lstrip()
        else:
            # Shouldn't happen in valid syntax
            return [line] + continuation_lines

        body, comment = split_comment(body_with_comment)
        body = body.rstrip()

        # Build continuation body
        if continuation_lines:
            full_body = body
            if full_body:
                full_body += "\n"
            full_body += "\n".join(continuation_lines)
        else:
            full_body = body

        # Build output lines
        result = []

        if has_types or ret_type:
            # Build type signature
            param_types = []
            param_names = []
            for name, typ in params:
                param_names.append(name)
                if typ:
                    param_types.append(typ)

            if param_types and ret_type:
                if len(param_types) == 1:
                    type_sig = f"{param_types[0]} -> {ret_type}"
                else:
                    type_sig = " * ".join(param_types) + f" -> {ret_type}"
                result.append(f"{indent}{func_name} : {type_sig}")
            elif ret_type and not param_types:
                # Zero-param function with return type: def f(): RET = body
                result.append(f"{indent}{func_name} : {ret_type}")
            elif param_types and not ret_type:
                # Has param types but no return type (unusual but handle)
                pass  # No type signature

            # Build definition line
            param_list = ", ".join(param_names)
            def_line = f"{indent}{func_name}({param_list}) ="
            if full_body:
                if "\n" in full_body:
                    def_line += "\n" + full_body
                else:
                    def_line += " " + full_body
            if comment:
                if "\n" not in full_body:
                    def_line += "  " + comment
            result.append(def_line)
        else:
            # No types at all, just remove 'def'
            param_list = ", ".join([p[0] for p in params])
            def_line = f"{indent}{func_name}({param_list}) ="
            if full_body:
                if "\n" in full_body:
                    def_line += "\n" + full_body
                else:
                    def_line += " " + full_body
            if comment:
                if "\n" not in full_body:
                    def_line += "  " + comment
            result.append(def_line)

        return result

    # Try typed variable: name: TYPE = body or name = body
    m = re.match(r"(\w+)\s*:\s*", rest)
    if m:
        var_name = m.group(1)
        after_name = rest[m.end() :]
        # Find = sign
        eq_idx = after_name.find("=")
        if eq_idx >= 0:
            type_str = after_name[:eq_idx].rstrip()
            body_with_comment = after_name[eq_idx + 1 :].lstrip()
            body, comment = split_comment(body_with_comment)
            body = body.rstrip()

            if continuation_lines:
                full_body = body
                if full_body:
                    full_body += "\n"
                full_body += "\n".join(continuation_lines)
            else:
                full_body = body

            result = []
            result.append(f"{indent}{var_name} : {type_str}")
            def_line = f"{indent}{var_name} ="
            if full_body:
                if "\n" in full_body:
                    def_line += "\n" + full_body
                else:
                    def_line += " " + full_body
            if comment:
                if "\n" not in full_body:
                    def_line += "  " + comment
            result.append(def_line)
            return result

    # Simple variable: name = body
    m = re.match(r"(\w+)\s*=\s*", rest)
    if m:
        var_name = m.group(1)
        body_with_comment = rest[m.end() :]
        body, comment = split_comment(body_with_comment)
        body = body.rstrip()

        if continuation_lines:
            full_body = body
            if full_body:
                full_body += "\n"
            full_body += "\n".join(continuation_lines)
        else:
            full_body = body

        def_line = f"{indent}{var_name} ="
        if full_body:
            if "\n" in full_body:
                def_line += "\n" + full_body
            else:
                def_line += " " + full_body
        if comment:
            if "\n" not in full_body:
                def_line += "  " + comment
        return [def_line]

    # Fallback: just remove 'def '
    return [indent + rest] + continuation_lines


def is_continuation_line(line):
    """Check if a line is a continuation of the previous def."""
    stripped = line.lstrip()
    if not stripped:
        return False
    # Continuation lines are indented and don't start with def, comments, or top-level constructs
    if line[0] in (" ", "\t") and not stripped.startswith("def "):
        if not stripped.startswith("(*"):
            return True
        # (* RUN: and (* CHECK: are directives, not continuations
        if stripped.startswith("(* RUN:") or stripped.startswith("(* CHECK"):
            return False
        return True
    return False


def convert_file(content):
    """Convert entire file content from def syntax to new syntax."""
    lines = content.split("\n")
    result = []
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()

        # Check if this is a def line
        if stripped.startswith("def "):
            # Collect continuation lines
            continuations = []
            j = i + 1
            while j < len(lines) and is_continuation_line(lines[j]):
                continuations.append(lines[j])
                j += 1

            converted = convert_def_line(line, continuations)
            result.extend(converted)
            i = j
        else:
            result.append(line)
            i += 1

    return "\n".join(result)


def main():
    if len(sys.argv) < 2:
        print("Usage: convert-def-syntax.py FILE [--inplace]", file=sys.stderr)
        sys.exit(1)

    filepath = sys.argv[1]
    inplace = "--inplace" in sys.argv

    with open(filepath, "r") as f:
        content = f.read()

    converted = convert_file(content)

    if inplace:
        with open(filepath, "w") as f:
            f.write(converted)
    else:
        print(converted, end="")


if __name__ == "__main__":
    main()
