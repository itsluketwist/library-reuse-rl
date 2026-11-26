"""Statically find the libraries a python code block imports and actually uses."""

# lets the `ast.TryStar` annotation below work on python 3.10, which doesn't have it
from __future__ import annotations

import ast
import sys
import warnings
from collections.abc import Iterable
from dataclasses import dataclass


# every standard-library module name for the running python version
STDLIB_MODULES: frozenset[str] = frozenset(sys.stdlib_module_names)

# names that stand for code the response doesn't have, not a real library: a placeholder for
# the reader's own module ("your_module", "my_utils"), or a generic name for one of a project's
# own files ("utils", "models"). a model that writes `from your_module import parse` beside its
# own code is not reusing a library, so by default these never count as external (see
# is_placeholder_module). every name is lowercase, and matched case-insensitively
PLACEHOLDER_PREFIXES: tuple[str, ...] = ("your_", "my_", "some_", "example_")
PLACEHOLDER_SUFFIXES: tuple[str, ...] = (
    "_utils",
    "_util",
    "_helper",
    "_helpers",
    "_config",
    "_module",
    "_service",
    "_services",
)
PLACEHOLDER_NAMES: frozenset[str] = frozenset(
    {
        "api",
        "app",
        "cli",
        "common",
        "config",
        "core",
        "database",
        "db",
        "helpers",
        "lib",
        "main",
        "models",
        "module",
        "mymodule",
        "project",
        "services",
        "settings",
        "src",
        "tasks",
        "utils",
    },
)


def is_placeholder_module(
    name: str,
) -> bool:
    """
    Check whether a top-level module name is a placeholder rather than a real library.

    A placeholder is a generic project-module name (PLACEHOLDER_NAMES, e.g. `utils`), or a
    name with a placeholder prefix or suffix (e.g. `your_module`, `data_utils`).

    Returns True for a placeholder name.
    """
    lowered = name.lower()
    return (
        lowered in PLACEHOLDER_NAMES
        or lowered.startswith(PLACEHOLDER_PREFIXES)
        or lowered.endswith(PLACEHOLDER_SUFFIXES)
    )


@dataclass(frozen=True)
class CodeBlock:
    """
    One python code block and the library evidence found in it, without running it.

    A library is "imported" when the block imports it, and "used" when the block then
    actually references a name bound by that import (e.g. `pd.read_csv` after
    `import pandas as pd`). Names are top-level import names with their original case
    (e.g. `PIL`, not `pil`). Standard-library modules and the project's own local modules
    are reported separately and never counted as external. Placeholder modules (see
    is_placeholder_module) are left out of the external ones too, unless asked otherwise,
    and listed in `placeholder_imports`.
    """

    code: str
    valid: bool
    error: str | None
    external_imports: frozenset[str]
    external_used: frozenset[str]
    stdlib_imports: frozenset[str]
    stdlib_used: frozenset[str]
    placeholder_imports: frozenset[str] = frozenset()


class _ImportUsageVisitor(ast.NodeVisitor):
    """
    Walk a syntax tree, tracking which names are bound to imported libraries.

    Each scope (module, function, class, lambda, comprehension) maps a name to the set of
    libraries it may refer to, which is empty once the name is rebound to something else.
    This shadowing check stops `pd = object(); pd.x` counting as using pandas. Alternative
    paths (if/else, try/except) are visited separately and their bindings merged, so a
    fallback like `except ImportError: np = None` doesn't erase the import.
    """

    def __init__(
        self,
        local_modules: set[str],
    ) -> None:
        self.local_modules = local_modules
        self.scopes: list[dict[str, frozenset[str]]] = [{}]
        self.imported: set[str] = set()
        self.used: set[str] = set()

    def _is_library(
        self,
        root: str,
    ) -> bool:
        """
        Check whether a top-level module name is a library we track.

        Returns False for the project's own modules and for `__future__`.
        """
        return bool(root) and root not in self.local_modules and root != "__future__"

    def _bind(
        self,
        name: str,
        root: str,
    ) -> None:
        """Bind a name in the current scope to the library it was imported from."""
        self.scopes[-1][name] = frozenset({root})
        if self._is_library(root):
            self.imported.add(root)

    def _shadow(
        self,
        name: str,
    ) -> None:
        """Rebind a name in the current scope to something that is not a library."""
        self.scopes[-1][name] = frozenset()

    def _resolve(
        self,
        name: str,
    ) -> frozenset[str]:
        """
        Look a name up from the innermost scope outwards.

        Returns the libraries the name may refer to, empty if it isn't bound to one.
        """
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        return frozenset()

    def _snapshot(
        self,
    ) -> list[dict[str, frozenset[str]]]:
        """
        Copy the current scopes, so a branch can start from (or return to) this point.

        Returns the copied scopes.
        """
        return [scope.copy() for scope in self.scopes]

    @staticmethod
    def _join(
        states: list[list[dict[str, frozenset[str]]]],
    ) -> list[dict[str, frozenset[str]]]:
        """
        Merge the scopes left by several alternative paths, keeping every possible library.

        Returns the merged scopes.
        """
        # every path is at the same scope depth, since scopes are popped before a path ends
        assert states and all(len(state) == len(states[0]) for state in states)
        joined = []
        for index in range(len(states[0])):
            names = set().union(*(state[index] for state in states))
            joined.append(
                {
                    name: frozenset().union(
                        *(state[index].get(name, frozenset()) for state in states),
                    )
                    for name in names
                },
            )
        return joined

    def _path(
        self,
        nodes: Iterable[ast.AST],
        entry: list[dict[str, frozenset[str]]],
    ) -> list[dict[str, frozenset[str]]]:
        """
        Visit one alternative path, starting from a copy of the entry scopes.

        Returns the scopes at the end of the path.
        """
        self.scopes = [scope.copy() for scope in entry]
        for node in nodes:
            self.visit(node)
        return self._snapshot()

    # --- alternative paths, which must not leak bindings into one another ---

    def visit_If(
        self,
        node: ast.If,
    ) -> None:
        self.visit(node.test)
        entry = self._snapshot()
        self.scopes = self._join(
            [
                self._path(node.body, entry),
                self._path(node.orelse, entry),
            ],
        )

    def visit_IfExp(
        self,
        node: ast.IfExp,
    ) -> None:
        self.visit(node.test)
        entry = self._snapshot()
        self.scopes = self._join(
            [
                self._path([node.body], entry),
                self._path([node.orelse], entry),
            ],
        )

    def visit_BoolOp(
        self,
        node: ast.BoolOp,
    ) -> None:
        # `a or b` may stop after any value, so each value is a possible end point
        exits = []
        for value in node.values:
            self.visit(value)
            exits.append(self._snapshot())
        self.scopes = self._join(exits)

    @staticmethod
    def _can_raise(
        node: ast.AST,
    ) -> bool:
        """
        Check whether a statement could raise, roughly - only trivial ones are ruled out.

        Returns False for `pass`, `global`, `nonlocal`, a bare constant, or a constant
        assigned to plain names, and True otherwise.
        """
        if isinstance(node, (ast.Pass, ast.Global, ast.Nonlocal)):
            return False
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            return False
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Constant)
            and all(isinstance(target, ast.Name) for target in node.targets)
        ):
            return False
        return True

    def visit_Try(
        self,
        node: ast.Try | ast.TryStar,
    ) -> None:
        # the scopes at every point the try body could raise, where a handler may start
        exceptional = []
        for statement in node.body:
            if isinstance(statement, (ast.Import, ast.ImportFrom)):
                # `import a, b` binds left to right, so can fail after binding `a`
                for alias in statement.names:
                    exceptional.append(self._snapshot())
                    if isinstance(statement, ast.Import):
                        self.visit_Import(ast.Import(names=[alias]))
                    else:
                        self.visit_ImportFrom(
                            ast.ImportFrom(
                                module=statement.module,
                                names=[alias],
                                level=statement.level,
                            ),
                        )
            else:
                if self._can_raise(statement):
                    exceptional.append(self._snapshot())
                self.visit(statement)
                if isinstance(
                    statement,
                    (
                        ast.If,
                        ast.For,
                        ast.AsyncFor,
                        ast.While,
                        ast.With,
                        ast.AsyncWith,
                        ast.Try,
                    ),
                ):
                    # a compound statement can rebind names and then raise
                    exceptional.append(self._snapshot())

        # the else block runs after a body that didn't raise, each handler after one that did
        body_exit = self._snapshot()
        normal_exit = self._path(node.orelse, body_exit)
        handler_entry = self._join(exceptional) if exceptional else body_exit
        exits = [normal_exit]
        for handler in node.handlers:
            handler_exit = self._path([handler], handler_entry)
            exits.append(handler_exit)
            if sys.version_info >= (3, 11) and isinstance(node, ast.TryStar):
                # several `except*` handlers can run, so a later one may see earlier bindings
                handler_entry = self._join([handler_entry, handler_exit])
        self.scopes = self._join(exits)
        for statement in node.finalbody:
            self.visit(statement)

    visit_TryStar = visit_Try

    def _shadow_target(
        self,
        target: ast.expr,
    ) -> None:
        """Shadow every plain name assigned by a target (e.g. `a, (b, c) = ...`)."""
        if isinstance(target, ast.Name):
            self._shadow(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                self._shadow_target(element)
        elif isinstance(target, ast.Starred):
            self._shadow_target(target.value)

    # --- imports ---

    def visit_Import(
        self,
        node: ast.Import,
    ) -> None:
        for alias in node.names:
            # `import a.b` binds `a`, `import a.b as c` binds `c` - both refer to library `a`
            root = alias.name.split(".", maxsplit=1)[0]
            self._bind(alias.asname or root, root)

    def visit_ImportFrom(
        self,
        node: ast.ImportFrom,
    ) -> None:
        # relative imports (`from . import x`) are always the project's own code
        if node.level or not node.module:
            return
        root = node.module.split(".", maxsplit=1)[0]
        if self._is_library(root):
            self.imported.add(root)
        for alias in node.names:
            # a star import binds unknown names, so its usage can't be tracked
            if alias.name != "*":
                self._bind(alias.asname or alias.name, root)

    # --- usage ---

    def visit_Name(
        self,
        node: ast.Name,
    ) -> None:
        if isinstance(node.ctx, ast.Load):
            self.used.update(
                root for root in self._resolve(node.id) if self._is_library(root)
            )

    # --- assignments, which shadow previously imported names ---

    def visit_Assign(
        self,
        node: ast.Assign,
    ) -> None:
        self.visit(node.value)
        for target in node.targets:
            self.visit(target)
            self._shadow_target(target)

    def visit_AnnAssign(
        self,
        node: ast.AnnAssign,
    ) -> None:
        if node.value is not None:
            self.visit(node.value)
        self.visit(node.annotation)
        self.visit(node.target)
        self._shadow_target(node.target)

    def visit_AugAssign(
        self,
        node: ast.AugAssign,
    ) -> None:
        self.visit(node.value)
        self.visit(node.target)
        self._shadow_target(node.target)

    def visit_NamedExpr(
        self,
        node: ast.NamedExpr,
    ) -> None:
        self.visit(node.value)
        self._shadow_target(node.target)

    def visit_Delete(
        self,
        node: ast.Delete,
    ) -> None:
        for target in node.targets:
            self.visit(target)
            self._shadow_target(target)

    def visit_For(
        self,
        node: ast.For | ast.AsyncFor,
    ) -> None:
        self.visit(node.iter)
        self._shadow_target(node.target)
        for statement in [*node.body, *node.orelse]:
            self.visit(statement)

    visit_AsyncFor = visit_For

    def visit_With(
        self,
        node: ast.With | ast.AsyncWith,
    ) -> None:
        for item in node.items:
            self.visit(item.context_expr)
            if item.optional_vars is not None:
                self._shadow_target(item.optional_vars)
        for statement in node.body:
            self.visit(statement)

    visit_AsyncWith = visit_With

    def visit_ExceptHandler(
        self,
        node: ast.ExceptHandler,
    ) -> None:
        if node.type is not None:
            self.visit(node.type)
        if node.name is not None:
            self._shadow(node.name)
        for statement in node.body:
            self.visit(statement)
        if node.name is not None:
            # python deletes the exception name when the handler ends
            self._shadow(node.name)

    # --- new scopes ---

    def _visit_arguments(
        self,
        arguments: ast.arguments,
    ) -> dict[str, frozenset[str]]:
        """
        Visit argument defaults and annotations, which belong to the enclosing scope.

        Returns a new scope with every argument name shadowed.
        """
        for default in [*arguments.defaults, *arguments.kw_defaults]:
            if default is not None:
                self.visit(default)

        all_arguments = [
            *arguments.posonlyargs,
            *arguments.args,
            *arguments.kwonlyargs,
        ]
        if arguments.vararg is not None:
            all_arguments.append(arguments.vararg)
        if arguments.kwarg is not None:
            all_arguments.append(arguments.kwarg)

        scope: dict[str, frozenset[str]] = {}
        for argument in all_arguments:
            if argument.annotation is not None:
                self.visit(argument.annotation)
            scope[argument.arg] = frozenset()
        return scope

    def visit_FunctionDef(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        if node.returns is not None:
            self.visit(node.returns)
        function_scope = self._visit_arguments(node.args)
        self._shadow(node.name)

        self.scopes.append(function_scope)
        for statement in node.body:
            self.visit(statement)
        self.scopes.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(
        self,
        node: ast.Lambda,
    ) -> None:
        lambda_scope = self._visit_arguments(node.args)
        self.scopes.append(lambda_scope)
        self.visit(node.body)
        self.scopes.pop()

    def visit_ClassDef(
        self,
        node: ast.ClassDef,
    ) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        for base in node.bases:
            self.visit(base)
        for keyword in node.keywords:
            self.visit(keyword.value)
        self._shadow(node.name)

        self.scopes.append({})
        for statement in node.body:
            self.visit(statement)
        self.scopes.pop()

    def _visit_comprehension(
        self,
        generators: list[ast.comprehension],
        results: list[ast.expr],
    ) -> None:
        """Visit a comprehension, whose loop variables live in their own scope."""
        self.scopes.append({})
        for generator in generators:
            self.visit(generator.iter)
            self._shadow_target(generator.target)
            for condition in generator.ifs:
                self.visit(condition)
        for result in results:
            self.visit(result)
        self.scopes.pop()

    def visit_ListComp(
        self,
        node: ast.ListComp,
    ) -> None:
        self._visit_comprehension(node.generators, [node.elt])

    def visit_SetComp(
        self,
        node: ast.SetComp,
    ) -> None:
        self._visit_comprehension(node.generators, [node.elt])

    def visit_GeneratorExp(
        self,
        node: ast.GeneratorExp,
    ) -> None:
        self._visit_comprehension(node.generators, [node.elt])

    def visit_DictComp(
        self,
        node: ast.DictComp,
    ) -> None:
        self._visit_comprehension(node.generators, [node.key, node.value])


def parse_python(
    code: str,
) -> ast.Module:
    """
    Parse python source into a syntax tree, without python's style warnings.

    Model-written code often has style mistakes, like an invalid escape sequence in a
    normal string, which make python emit a SyntaxWarning. They don't affect validity, and
    would flood the logs.

    Returns the parsed syntax tree, or raises SyntaxError when the code is invalid.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        return ast.parse(code)


def analyse_code(
    code: str,
    local_modules: Iterable[str] = (),
    ignore_placeholders: bool = True,
) -> CodeBlock:
    """
    Parse a python code block and find the libraries it imports and uses, without
    running it.

    `local_modules` are the top-level names of the project's own modules (e.g. `app` for
    `from app.models import User`), which are never counted as libraries. Relative imports
    are always treated as local. With `ignore_placeholders` (the default), placeholder
    modules such as `your_module` or `utils` aren't counted as libraries either, and are
    listed in `placeholder_imports` instead. Code that fails to parse is reported as
    invalid, with no libraries.

    Returns the analysed CodeBlock.
    """
    visitor = _ImportUsageVisitor(local_modules=set(local_modules))
    try:
        visitor.visit(parse_python(code))
    except (SyntaxError, ValueError, RecursionError) as error:
        # ValueError covers source containing null bytes, and RecursionError covers
        # extremely deeply nested code - a reward function must never crash on a response
        return CodeBlock(
            code=code,
            valid=False,
            error=f"{type(error).__name__}: {error}",
            external_imports=frozenset(),
            external_used=frozenset(),
            stdlib_imports=frozenset(),
            stdlib_used=frozenset(),
        )

    # split every library into the standard library and external (third-party) ones, setting
    # aside placeholder modules, which stand for code the response doesn't have
    external = visitor.imported - STDLIB_MODULES
    placeholders = (
        {name for name in external if is_placeholder_module(name)}
        if ignore_placeholders
        else set()
    )
    return CodeBlock(
        code=code,
        valid=True,
        error=None,
        external_imports=frozenset(external - placeholders),
        external_used=frozenset(visitor.used - STDLIB_MODULES - placeholders),
        stdlib_imports=frozenset(visitor.imported & STDLIB_MODULES),
        stdlib_used=frozenset(visitor.used & STDLIB_MODULES),
        placeholder_imports=frozenset(placeholders),
    )
