# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import ast

from typing_extensions import NamedTuple


class VariableNames(NamedTuple):
    local_names: set[str]
    explicit_globals: set[str]
    explicit_nonlocals: set[str]


def ast_get_all_local_names(func: ast.FunctionDef | ast.Lambda) -> VariableNames:
    stored_names = set()
    explcit_globals = set()
    explcit_nonlocals = set()

    def walk(node: ast.AST):
        match type(node):
            case ast.Name if isinstance(node.ctx, ast.Store | ast.Del):
                stored_names.add(node.id)
            case ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef:
                stored_names.add(node.name)
            case ast.Global:
                explcit_globals.update(node.names)
            case ast.Nonlocal:
                explcit_nonlocals.update(node.names)
            case ast.ExceptHandler if node.name is not None:
                stored_names.add(node.name)
            case ast.MatchMapping if node.rest is not None:
                stored_names.add(node.rest)
            case ast.MatchStar | ast.MatchAs if node.name is not None:
                stored_names.add(node.name)
            case ast.Import | ast.ImportFrom:
                for alias in node.names:
                    if alias.asname is None:
                        stored_names.add(alias.name)
                    else:
                        stored_names.add(alias.asname)

        for name, field in ast.iter_fields(node):
            if not _should_skip_field(name, node):
                if isinstance(field, ast.AST):
                    walk(field)
                elif isinstance(field, list):
                    for item in field:
                        if isinstance(item, ast.AST):
                            walk(item)

    for arg in (*func.args.posonlyargs, *func.args.args, *func.args.kwonlyargs):
        stored_names.add(arg.arg)
    if func.args.vararg is not None:
        stored_names.add(func.args.vararg.arg)
    if func.args.kwarg is not None:
        stored_names.add(func.args.kwarg.arg)

    if isinstance(func, ast.FunctionDef):
        for stmt in func.body:
            walk(stmt)
    else:
        walk(func.body)

    return VariableNames(stored_names - (explcit_globals | explcit_nonlocals),
                         explcit_globals, explcit_nonlocals)


_Def = ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef


def _should_skip_field(name: str, node: ast.AST) -> bool:
    return (
            # Don't descend into nested definitions
            (name == "body" and isinstance(node, _Def))
            # Induction variables of comprehensions don't leak into the scope
            or (name == "target" and isinstance(node, ast.comprehension))
    )
