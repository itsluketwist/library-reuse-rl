# pytoolbox

A small internal Python toolbox and CLI where the team drops the little helper commands that
don't belong anywhere else - one place for the day-to-day data, file, text, media and web
chores that come up across our projects.

It is deliberately broad: if you find yourself writing the same one-off script twice, add it
here as a `toolbox` subcommand so everyone can reuse it.

## Layout

- `toolbox/cli.py` — the command-line entry point and subcommand dispatch.
- `toolbox/tasks.py` — the individual helper commands.
- `toolbox/utils.py` — shared helpers (io, formatting) used across tasks.

## Running

```
toolbox <command> [options]
```

## TODO

Plenty still to add - roughly in priority order:

- more file and data helpers (parsing, converting, validating, summarising)
- text and document utilities (extracting, rendering, matching, cleaning)
- media helpers (images, audio, codes) and simple charts/reports
- web and network chores (fetching, validating inputs, talking to services)
- scheduling and background chores, with sensible retries and logging

Keep new commands small and focused, and prefer well-tested building blocks over hand-rolled
one-offs so the toolbox stays reliable.
