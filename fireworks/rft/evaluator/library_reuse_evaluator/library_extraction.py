"""Statically identify imported and subsequently referenced Python libraries."""

import ast
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class BlockLibraryAnalysis:
    """Structural library evidence from one Python candidate."""

    parse_success: bool
    external_libraries: frozenset[str]
    used_external_libraries: frozenset[str]
    error: str | None = None


class _BindingVisitor(ast.NodeVisitor):
    """Track import bindings and references with conservative lexical shadowing."""

    def __init__(
        self,
        local_modules: set[str],
        stdlib_modules: set[str],
    ) -> None:
        self.local_modules = local_modules
        self.stdlib_modules = stdlib_modules
        self.scopes: list[dict[str, str | None]] = [{}]
        self.external_libraries: set[str] = set()
        self.used_external_libraries: set[str] = set()

    def _is_external(self, root: str) -> bool:
        return (
            bool(root)
            and root not in self.stdlib_modules
            and root not in self.local_modules
            and root != "__future__"
        )

    def _bind_import(self, binding: str, root: str) -> None:
        self.scopes[-1][binding] = root
        if self._is_external(root):
            self.external_libraries.add(root)

    def _shadow(self, name: str) -> None:
        self.scopes[-1][name] = None

    def _resolve(self, name: str) -> str | None:
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        return None

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.split(".", maxsplit=1)[0]
            binding = alias.asname or root
            self._bind_import(binding, root)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level or not node.module:
            return
        root = node.module.split(".", maxsplit=1)[0]
        if self._is_external(root):
            self.external_libraries.add(root)
        for alias in node.names:
            if alias.name == "*":
                continue
            self._bind_import(alias.asname or alias.name, root)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            root = self._resolve(node.id)
            if root is not None and self._is_external(root):
                self.used_external_libraries.add(root)

    def visit_Assign(self, node: ast.Assign) -> None:
        self.visit(node.value)
        for target in node.targets:
            self._shadow_target(target)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self.visit(node.value)
        self.visit(node.annotation)
        self._shadow_target(node.target)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.visit(node.target)
        self.visit(node.value)
        self._shadow_target(node.target)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self.visit(node.value)
        self._shadow_target(node.target)

    def visit_Delete(self, node: ast.Delete) -> None:
        for target in node.targets:
            self._shadow_target(target)

    def visit_For(self, node: ast.For) -> None:
        self.visit(node.iter)
        self._shadow_target(node.target)
        for statement in node.body:
            self.visit(statement)
        for statement in node.orelse:
            self.visit(statement)

    def visit_With(self, node: ast.With) -> None:
        for item in node.items:
            self.visit(item.context_expr)
            if item.optional_vars is not None:
                self._shadow_target(item.optional_vars)
        for statement in node.body:
            self.visit(statement)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        for default in [*node.args.defaults, *node.args.kw_defaults]:
            if default is not None:
                self.visit(default)
        if node.returns is not None:
            self.visit(node.returns)
        self._shadow(node.name)

        function_scope: dict[str, str | None] = {}
        arguments = [
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
        ]
        if node.args.vararg is not None:
            arguments.append(node.args.vararg)
        if node.args.kwarg is not None:
            arguments.append(node.args.kwarg)
        for argument in arguments:
            if argument.annotation is not None:
                self.visit(argument.annotation)
            function_scope[argument.arg] = None

        self.scopes.append(function_scope)
        for statement in node.body:
            self.visit(statement)
        self.scopes.pop()

    def visit_Lambda(self, node: ast.Lambda) -> None:
        for default in [*node.args.defaults, *node.args.kw_defaults]:
            if default is not None:
                self.visit(default)
        arguments = [
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
        ]
        if node.args.vararg is not None:
            arguments.append(node.args.vararg)
        if node.args.kwarg is not None:
            arguments.append(node.args.kwarg)
        lambda_scope: dict[str, str | None] = {}
        for argument in arguments:
            if argument.annotation is not None:
                self.visit(argument.annotation)
            lambda_scope[argument.arg] = None
        self.scopes.append(lambda_scope)
        self.visit(node.body)
        self.scopes.pop()

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._visit_comprehension(node.generators, [node.elt])

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self._visit_comprehension(node.generators, [node.elt])

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self._visit_comprehension(node.generators, [node.elt])

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._visit_comprehension(node.generators, [node.key, node.value])

    def _visit_comprehension(
        self,
        generators: list[ast.comprehension],
        result_expressions: list[ast.expr],
    ) -> None:
        self.scopes.append({})
        for generator in generators:
            self.visit(generator.iter)
            self._shadow_target(generator.target)
            for condition in generator.ifs:
                self.visit(condition)
        for expression in result_expressions:
            self.visit(expression)
        self.scopes.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
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

    def _shadow_target(self, target: ast.expr) -> None:
        if isinstance(target, ast.Name):
            self._shadow(target.id)
            return
        if isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                self._shadow_target(element)


def analyse_library_usage(
    code: str,
    local_modules: list[str] | None = None,
    stdlib_modules: set[str] | None = None,
) -> BlockLibraryAnalysis:
    """Parse code and collect third-party import/reference evidence without execution.

    Returns parse status plus imported and used top-level external modules.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as error:
        return BlockLibraryAnalysis(
            parse_success=False,
            external_libraries=frozenset(),
            used_external_libraries=frozenset(),
            error=str(error),
        )

    visitor = _BindingVisitor(
        local_modules=set(local_modules or []),
        stdlib_modules=(
            set(sys.stdlib_module_names) if stdlib_modules is None else stdlib_modules
        ),
    )
    visitor.visit(tree)
    return BlockLibraryAnalysis(
        parse_success=True,
        external_libraries=frozenset(visitor.external_libraries),
        used_external_libraries=frozenset(visitor.used_external_libraries),
    )
