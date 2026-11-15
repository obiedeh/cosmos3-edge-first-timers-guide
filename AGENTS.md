# AGENTS.md: cosmos3-edge-first-timers-guide

## Purpose

A Google Colab how-to for the NVIDIA Cosmos3-Edge reasoner on images and video,
with every setting measured for its effect on accuracy and response time.

## Layout

```
cosmos3_edge_first_timers_guide.ipynb    the notebook, no outputs
notebook_src.txt                          the cells as plain text; edit here
build_notebook.py                         rebuilds the notebook from the source
records/                                  executed copies, named by board and date
records/server_tests/                     the vLLM requests behind Part 17 and their output
docs/figures/                             charts shown in the README, taken from a record
tests/                                    source-sync and repository hygiene tests
```

## Rules

- Edit `notebook_src.txt`, then run `python build_notebook.py`. Never edit the
  notebook file by hand.
- The committed notebook carries no outputs. Executed copies go in `records/`.
- Results in the README are measured values with board, date and record.
- Sample media, weights and caches never enter git.
- Every change adds a dated line to `CHANGELOG.md` in the same commit.

## Repository hygiene

Read and follow the [Repository hygiene policy in obiedeh-brand/AGENTS.md](https://github.com/obiedeh/obiedeh-brand/blob/main/AGENTS.md#repository-hygiene-standing-every-repo), the single source of these rules.

- Name boards only: `jetson-agx-thor`, `jetson-orin-nx`, `rtx-5090-workstation`.
- No host names, home-directory paths or LAN addresses in tracked prose, logs, commit messages or branch names; write commands against `~`.
- Never bypass or disable hooks.
- Keep handoff notes, reviews and plans outside the repository in `~/handoff_notes/`.
