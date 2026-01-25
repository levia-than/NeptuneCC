# -*- Python -*-

import os
import platform
import re
import subprocess
import tempfile

import lit.formats
import lit.util

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst
from lit.llvm.subst import FindTool


config.name = "neptune-tests"
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)
config.suffixes = ['.mlir']

config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.neptune_build_dir, 'test')
config.substitutions.append(("%PATH%", config.environment["PATH"]))
config.substitutions.append(("%shlibext", config.llvm_shlib_ext))

tool_dirs = [config.neptune_tools_dir, config.llvm_tools_dir]
tools = ["neptune-opt", "mlir-translate", "not", "FileCheck"]
llvm_config.add_tool_substitutions(tools, tool_dirs)
