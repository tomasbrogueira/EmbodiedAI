# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from .._ir import ir
from .._ir.core_ops import Assign


def eliminate_assign_ops(root_block: ir.Block):
    mapper = ir.Mapper(root_block.ctx)

    def walk(block):
        new_ops = []
        for op in block:
            op.remap_operands(mapper)
            if isinstance(op, Assign):
                mapper.set_var(op.result_var, op.value)
            else:
                for nested_block in op.nested_blocks:
                    walk(nested_block)
                new_ops.append(op)
        block[:] = new_ops

    walk(root_block)
