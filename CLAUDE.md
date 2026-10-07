# FinAI-Agent

Autonomous portfolio and investment research agent: LangGraph agent with RAG over SEC 10-K filings (Qdrant), live market data (yfinance), a verifier node, a guarded FastAPI streaming API, deployed to AWS EKS.

## Context

`docs/CONTEXT.md` records build state, library behaviour verified against installed versions, and decisions that depart from `PLAN.md`. Read it before resuming work.

## The plan

The full build plan is in `PLAN.md`. Task IDs look like `1.2.7`.

- When I give you a task ID, read that step in `PLAN.md` (the whole `### x.y` step, not only the task line) before doing anything.
- Work on one task ID at a time. Do not start the next task until the current one passes the step's **Done when** check.
- Before coding, restate the task, list the files you will create or change, and ask if anything is ambiguous.
- After the check passes, show its output, tick the task's checkbox in `PLAN.md` (`- [ ]` → `- [x]`), and commit with a message like `feat(1.2): chunk 10-K sections with metadata`.
- Tasks marked `[HUMAN]` need me (accounts, payments, secrets, console clicks). Stop and ask.

## Rules

- All configuration goes through the `Settings` class (pydantic-settings). No scattered `os.getenv`.
- Never commit secrets. `.env`, `data/`, `*.db`, `k8s/**/secret.yaml` stay gitignored.
- Pin versions (lockfile, Docker tags, Terraform providers). Never use `latest`.
- Every tool and module gets at least one pytest test.
- The LLM never does arithmetic; portfolio and ratio math happens in Python.
- `user_id` comes from auth or graph config, never from the request body or an LLM tool argument.
- Never invent APIs for fast-moving libraries (LangGraph, EdgarTools, LLM Guard, Ragas, FastEmbed). Check the installed version's docs or source first.
- Run `ruff check` and `ruff format` before each commit.

## Cost rules

- Never schedule nightly eval runs. Use Haiku as the eval judge.
- Never leave AWS resources running after a session; follow the teardown steps in `PLAN.md` (4.2.16–4.2.22).

## Commands

- Local services: `docker compose up -d`
- Tests: `uv run pytest`
- Lint: `uv run ruff check . && uv run ruff format .`
