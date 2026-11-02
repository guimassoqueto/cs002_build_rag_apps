# cs002_build_rag_apps

Build RAG Applications. The user writes course notes under `lessons/`; the
`lesson-script` skill writes the runnable script for each. If you can't load skills
and are asked for it, read `.agents/skills/lesson-script/SKILL.md` and follow it.

## Lesson folders

A lesson that ships extra files (a `compose.yaml`, a `Modelfile`, ...) gets a
folder named after its stem, holding the notes, the script and those files:

```
lessons/module02/03_qa_bot_with_local_qwen3/
    03_qa_bot_with_local_qwen3.md     # notes
    03_qa_bot_with_local_qwen3.py     # the script
    compose.yaml                      # local services the script talks to
    Modelfile
```

It is still a lesson script, not a self-contained project (no `pyproject.toml`): the
skill applies, it uses the root `.venv` and imports `app.config`. Model ids and service
URLs for local services go in `app/config/envs.py` too.

## Self-contained projects

A folder under `lessons/` with its own `pyproject.toml` (`lessons/module03/04_icebreaker_bot_with_local_qwen3/`)
is an independent project, not a lesson, so the skill doesn't apply. Follow its own
README and Makefile, use its own `.venv`, never import `app.config` from it or add its
dependencies to the root `pyproject.toml`, and add it to `extend-exclude` there.

## Conventions

- Python 3.14 from asdf (`.tool-versions`). uv manages `.venv`: add dependencies unpinned
  with `uv add --raw <pkg>` (plus `--dev` for tooling), run tools with `uv run`. Never
  `pip install`: `.venv` has no pip, so it would hit asdf's global Python.
- `make lint` / `make lint-check`: ruff fix / check only (line length 120, config in
  `pyproject.toml`).
- Don't commit unless asked. A commit-msg hook enforces conventional commits (`feat:`,
  `fix:`, `chore:`, ...); `make commit` runs commitizen.
- `output/` is gitignored: never commit generated files. `make clean-output` deletes them.
- Use `git mv` when moving tracked files.
- `README.md` is the human-facing doc; update it when the architecture changes.
- Skills live in `.agents/skills/<name>/`, so edit them there. `.claude/skills/<name>` is
  a relative symlink to each, for Claude Code.
