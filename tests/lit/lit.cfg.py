# -*- Python -*-

import os
import lit.formats
import lit.util

# Configuration file for the 'lit' test runner.

# Name of the test suite
config.name = "Polang"

# File extensions for test files
config.suffixes = [".po"]

# Test format - ShTest allows RUN: lines with shell commands
config.test_format = lit.formats.ShTest(not lit.util.which("bash"))

# Test source root is the directory containing this file
config.test_source_root = os.path.dirname(__file__)

# Test execution root (where tests run from)
config.test_exec_root = config.polang_obj_root

# Tool substitutions - these are replaced in RUN: lines
config.substitutions.append(("%polang_compiler", config.polang_compiler_path))
config.substitutions.append(("%polang_repl", config.polang_repl_path))
config.substitutions.append(("%FileCheck", config.filecheck_path))
config.substitutions.append(("%not", config.not_path))

# Features for conditional tests
if lit.util.which("bash"):
    config.available_features.add("shell")

# Environment setup
config.environment["PATH"] = os.pathsep.join(
    [os.path.dirname(config.polang_compiler_path), config.environment.get("PATH", "")]
)
