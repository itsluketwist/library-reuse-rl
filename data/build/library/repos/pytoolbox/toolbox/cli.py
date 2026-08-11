"""Command-line entry point for the toolbox - dispatches to the commands in tasks.py."""

import argparse

from toolbox import tasks


def build_parser() -> argparse.ArgumentParser:
    """Build the toolbox argument parser with its subcommands.

    Returns the parser.
    """
    parser = argparse.ArgumentParser(
        prog="toolbox", description="internal helper commands"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    count = subparsers.add_parser("count-lines", help="count the lines in a file")
    count.add_argument("path")

    summarise = subparsers.add_parser(
        "summarise", help="write a short summary of a file"
    )
    summarise.add_argument("path")
    summarise.add_argument("--out", required=True)

    return parser


def main() -> None:
    """Parse the command line and run the requested toolbox command."""
    args = build_parser().parse_args()

    if args.command == "count-lines":
        print(tasks.count_lines(args.path))
    elif args.command == "summarise":
        tasks.summarise_file(args.path, args.out)


if __name__ == "__main__":
    main()
