# Repository Guidelines

## Project Structure & Module Organization
Start from `Info.md` and `docs/plan.md` (the revised demo overview also lives in `BackgroundInfo.md`). Keep Python source in `src/` (pipelines, features, models, common). Store raw NASA downloads under gitignored `data/raw/`, derived tables in `data/processed/`, and trained artefacts in DVC/LFS-backed `artifacts/`. Place the web UI in `webapp/` (`app/`, `components/`, `public/`). Exploratory notebooks go in `notebooks/`; promote production-ready code into `src/`. Keep diagrams and ADRs in `docs/`.

## Build, Test, and Development Commands
We standardize on Python 3.11 via `uv` and Node 20.
- After editing `pyproject.toml`: `uv sync`.
- Validate data loaders: `uv run python -m src.pipelines.fetch_data --help`.
- Backend tests: `uv run pytest` (CI default runs `-m "not slow"`).
- Lint: `uv run ruff check .`.
- Inside `webapp/`: `npm install`, `npm run dev`, `npm run lint`, `npm run test` (Vitest) before pushing.

## Coding Style & Naming Conventions
Python uses `ruff` + `black` defaults (line length 120, 4-space indent) with type hints and NumPy-style docstrings that state units. Modules, functions, and files use `snake_case`; model checkpoints follow `mission_model-vX.pt`. React/TypeScript code keeps variables in `camelCase`, components in `PascalCase`, and SCSS modules as `Component.module.scss`. Notebooks stay light and export figures to `docs/figures/`.

## Testing Guidelines
Mirror package paths under `tests/` (e.g., `tests/pipelines/test_bls.py`) and tag expensive cases `@pytest.mark.slow`. The default CI leg runs `uv run pytest -m "not slow"`; schedule a nightly job that includes slow tests. Provide fixtures covering false positives and mission edge cases. UI tests live in `webapp/tests/` with Vitest snapshots for UI states. Target >=85% statement coverage and log evaluation metrics with each model PR.

## Commit & Pull Request Guidelines
History currently shows only `Initial commit`; adopt Conventional Commits (`type(scope): summary`) with descriptive bodies when behaviour changes. Reference issues and dataset versions touched. Pull requests must include a concise summary, relevant screenshots, current metrics, and a checklist confirming `uv run pytest` and `npm run test`. Mark WIP as Draft and request domain review when adjusting mission-specific heuristics.

## Data & Security Notes
Never commit raw FITS/CSV downloads; keep them in `data/raw/` via scripted fetches. Secrets belong in `.env.local` with a matching `.env.example`. Document new external services in `docs/integrations.md` and flag upstream schema changes in the weekly status notes.

---

## Scope and Agent Workflow
- Scope: this AGENTS.md applies to the entire repository. More specific rules in subfolders (if added later) take precedence there.
- Propose, then change: before large edits, outline a short plan (bulleted steps). Keep patches narrowly scoped.
- Keep edits surgical: avoid broad refactors unless accompanied by an ADR in `docs/adr/`.
- Tests first where practical: add/adjust minimal tests under `tests/` mirroring the package path.

## Adding a New Pipeline or Model
- Pipelines live under `src/pipelines/` with a `__main__.py` or a `main()` entry so they can be invoked via `python -m src.pipelines.<name>`.
- Models live under `src/models/`. Export trained artifacts to `artifacts/mission_model-vX/` and include the feature schema, calibrator, and version metadata (`model.json`).
- Provide a small fixture under `tests/fixtures/` (e.g., a trimmed light curve) to keep tests fast and deterministic.

## API and UI Stubs
- Backend: use FastAPI; contracts are documented in `docs/plan.md`. Put routes in `src/api/` (to be created) and expose a single ASGI app.
- Frontend: Next.js under `webapp/`. Keep charts client-side where possible; run all data and LLM calls server-side.

## LLM Usage Guidelines
- Provider keys must be read from server-side environment (e.g., `OPENAI_API_KEY`) and never exposed to the client.
- Explanations must be grounded in our own computed context (features, metrics, chart annotations, and mission helptext). If context is insufficient, respond that more evidence is needed.
- Log prompts minimally (hash or omit secrets); include an `explanation_id` to correlate responses with an evidence bundle.

## Data Governance
- Raw downloads (`data/raw/`) are never committed. Small, anonymized slices for tests may be stored under `tests/fixtures/`.
- Document each external table and API used in `docs/integrations.md`, including the schema fields we rely on and an example query.
- If a breaking upstream schema change is detected, open an issue and add a short note to weekly status.

## Conventional Commits (quick reference)
- `feat(pipelines): add KOI fetcher`
- `fix(models): calibrate probabilities with isotonic regression`
- `docs(plan): add API contracts`
- `test(bls): add odd/even ratio fixture`

## Definition of Done (per PR)
- Code compiles; `uv run ruff check .` passes; `uv run pytest -m "not slow"` passes locally.
- Updated or added docs where behavior changed.
- Artifacts (if updated) are versioned and small diffs are verified via LFS/DVC metadata.

