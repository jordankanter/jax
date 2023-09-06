# Copyright 2023 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial
import logging
import os
import pathlib
import platform
import sys

import jax._src.xla_bridge as xb

try:
  from jax._src.lib import gpu_plugin_extension as gpu_plugin_extension
except ImportError:
  gpu_plugin_extension = None
from jax._src.lib import xla_client
from jax._src.lib import xla_extension_version


logger = logging.getLogger(__name__)


def initialize():
  path = pathlib.Path(__file__).resolve().parent / "xla_cuda_plugin.so"
  if not path.exists():
    logger.warning(
        "WARNING: Native library %s does not exist. This most likely indicates"
        " an issue with how %s was built or installed.",
        path,
        __package__,
    )
  c_api = xb.register_plugin("cuda", priority=500, library_path=str(path))

  if gpu_plugin_extension and xla_extension_version >= 195:
    xla_client.register_custom_calls_and_handler_if_not_exist(
        "CUDA",
        partial(gpu_plugin_extension.register_gpu_custom_call_target, c_api),
    )
