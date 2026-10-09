# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from ._jax import (cutile_call, OutputPlaceholder, InputOutput,
                   register_ffi, register_primitive, HAS_JAX)
import logging
logger = logging.getLogger(__name__)

__all__ = [
    "cutile_call",
    "OutputPlaceholder",
    "InputOutput",
]

if HAS_JAX:
    register_ffi()
    register_primitive()
else:
    logger.warning("JAX is not available")
