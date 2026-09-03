"""Run Eval Protocol with a narrow fix for Literal-valued CLI options."""

import argparse
import importlib
import sys
import typing
from collections.abc import Callable
from typing import Any


utils = importlib.import_module("eval_protocol.cli_commands.utils")
_ORIGINAL_TYPE_CONVERTER = utils.argparse_type_from_hint


def _literal_value_parser(hint: Any) -> Callable[[str], Any] | None:
    """Build an argparse converter when a type hint is a Literal.

    Returns a validating converter, or ``None`` for a non-Literal hint.
    """
    unwrapped_hint = utils.unwrap_union(hint)
    if typing.get_origin(unwrapped_hint) is not typing.Literal:
        return None
    allowed_values = typing.get_args(unwrapped_hint)

    def parse_literal(value: str) -> Any:
        """Validate one command-line value against the Literal members.

        Returns the matching typed Literal member.
        """
        for allowed_value in allowed_values:
            if value == str(allowed_value):
                return allowed_value
        choices = ", ".join(repr(choice) for choice in allowed_values)
        raise argparse.ArgumentTypeError(
            f"expected one of {choices}; received {value!r}",
        )

    return parse_literal


def _compatible_type_converter(hint: Any) -> Any:
    """Handle Literal hints and delegate every other hint unchanged.

    Returns an argparse type converter.
    """
    return _literal_value_parser(hint) or _ORIGINAL_TYPE_CONVERTER(hint)


def main() -> None:
    """Apply the compatibility fix and run Eval Protocol's normal CLI."""
    # eval protocol 0.3.32 treats typing.Literal itself as the converter, which
    # rejects every value. patch only that converter before its parser is built.
    setattr(utils, "argparse_type_from_hint", _compatible_type_converter)

    eval_protocol_cli = importlib.import_module("eval_protocol.cli")
    sys.exit(eval_protocol_cli.main())


if __name__ == "__main__":
    main()
