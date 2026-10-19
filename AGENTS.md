# cs002_build_rag_apps

Build RAG Applications. The user writes course notes;

## The standing task

When the user points at a doc under `lessons/` ("do `module02/01_getting_started_with_gradio.md",
"here is a new doc"), read it and write the  script for it. The doc is
the spec - no need to ask what to build.

## Layout contract

Notes, script and generated files share one stem and live together:

```
lessons/module02/01_getting_started_with_gradio.md        # notes (the user writes these)
lessons/module02/01_getting_started_with_gradio.py        # the script (you write this)
lessons/module02/output/01_getting_started_with_gradio/   # what it generates (only if generates something) (gitignored)
    concepts.json
    message_types.csv
    index.html
```

- The script filename matches the doc filename exactly, `.md` -> `.py`, beside it.
- Module folders are zero padded: `module02`, never `module2`.
- A new module folder needs no configuration.

### Lesson folders

A lesson that ships extra files (a `compose.yaml`, a `Modelfile`, ...) gets a
folder named after its stem, holding the notes, the script and those files:

```
lessons/module02/03_qa_bot_with_local_qwen3/
    03_qa_bot_with_local_qwen3.md     # notes
    03_qa_bot_with_local_qwen3.py     # the script
    compose.yaml                      # local services the script talks to
    Modelfile
```

It is still a lesson script, not a self-contained project (no `pyproject.toml`):
all the rules here apply, it uses the root `.venv` and imports `app.config`.
Model ids and service URLs for local services go in `app/config/envs.py` too.

### Self-contained projects

A folder under `lessons/` that holds its own `pyproject.toml` (e.g.
`lessons/moduleNN/some_project/`) is an independent project, not a lesson
script. The rules above and below do not apply to it. Follow its own README and
Makefile, use its own `.venv`, and never import `app.config` from it or add its
dependencies to `devtools/requirements.txt`. Add each one to `extend-exclude` in
`devtools/pyproject.toml`.

## Writing a lesson script

Cover the concepts the doc actually describes, in the doc's order, one short
section per concept with a one-line comment introducing it. Flat top-level code:
no `main()`, no argparse, no try/except. These are read as examples, so they
should contain only the code the doc is about.

**Never hardcode a path or a model id.** Import them from `app/config/envs.py`:

| Need                          | Use                                                                   |
| ----------------------------- | --------------------------------------------------------------------- |
| the doc this script came from | `doc_path(__file__)`                                                  |
| somewhere to write a file     | `output_path(__file__, "concepts.json")` (parent directory is created) |
| which model                   | `DEFAULT_MODEL`, `DEFAULT_MAX_TOKENS`                                 |

Both path helpers derive everything from `__file__`, so renaming or moving a
lesson needs no edit anywhere.

Where results go:

- text, or anything you would just look at -> `print("[section label]", ...)`
- file shaped (`.json`, `.csv`, `.md`, ...) -> write through `output_path(...)`

The key loads itself: importing `app.config.envs` runs `load_dotenv()`, after
which `ChatAnthropic(model=DEFAULT_MODEL, max_tokens=DEFAULT_MAX_TOKENS)` finds
`ANTHROPIC_API_KEY` with no argument.

## Repo conventions

- Do not commit unless asked. A commit-msg hook enforces conventional commits
  (`feat:`, `fix:`, `chore:`, ...).
- `output/` is gitignored. Never commit generated files.
- Use `git mv` when moving tracked files.
- Python 3.14, venv at `.venv`, dependencies unpinned in `devtools/requirements.txt`.
- `README.md` is the human-facing doc; update it when the architecture changes.
