"""Statically find the libraries a python code block imports and actually uses."""

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

    Each scope (module, function, class, lambda, comprehension) maps a name to the library
    it was imported from, or to None once the name is rebound to something else. This
    shadowing check stops `pd = object(); pd.x` counting as using pandas.
    """

    def __init__(
        self,
        local_modules: set[str],
    ) -> None:
        self.local_modules = local_modules
        self.scopes: list[dict[str, str | None]] = [{}]
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
        self.scopes[-1][name] = root
        if self._is_library(root):
            self.imported.add(root)

    def _shadow(
        self,
        name: str,
    ) -> None:
        """Rebind a name in the current scope to something that is not a library."""
        self.scopes[-1][name] = None

    def _resolve(
        self,
        name: str,
    ) -> str | None:
        """
        Look a name up from the innermost scope outwards.

        Returns the library the name refers to, or None if it isn't bound to one.
        """
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        return None

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
            root = self._resolve(node.id)
            if root is not None and self._is_library(root):
                self.used.add(root)

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

    # --- new scopes ---

    def _visit_arguments(
        self,
        arguments: ast.arguments,
    ) -> dict[str, str | None]:
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

        scope: dict[str, str | None] = {}
        for argument in all_arguments:
            if argument.annotation is not None:
                self.visit(argument.annotation)
            scope[argument.arg] = None
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
