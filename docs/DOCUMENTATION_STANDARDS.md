# Documentation Standards

A consistent style makes docs easier to read, search, and maintain. Use this guide when writing or updating Markdown files in this repository.

## Structure
- Start with a single H1 (`# Title`)
- Use descriptive H2/H3 sections
- Prefer concise bullet lists over long paragraphs
- Keep code blocks short and focused

## Formatting
- Use backticks for commands, file paths, and code identifiers: `pip install -r requirements.txt`, `experiments/run_all_experiments.py`
- Use language-tagged code blocks for syntax highlighting:
  ```bash
  python experiments/train_yolov8.py --help
  ```
- Keep lists tight; avoid deep nesting
- Avoid heavy inline HTML; rely on Markdown features

## Links
- Use relative links for repository files: `[README](../README.md)`
- Verify external links periodically; prefer authoritative sources
- For images, store under `docs/images/` and link with relative paths

## Examples
- Provide runnable examples where possible
- Prefer commands that work from repo root
- Avoid environment-specific assumptions; document required env vars explicitly

## Versioning and History
- Keep `CHANGELOG.md` updated for user-facing changes
- Preserve historical context in `docs/legacy/` instead of deleting

## Style Consistency
- Use present tense and active voice
- Keep a friendly, direct tone
- Use parallel structure in lists; group related points together

## Checklist for New or Updated Docs
- Title and short overview present
- Accurate repository paths and commands
- Links are valid and relative where possible
- Code blocks have language tags
- Images saved in `docs/images/` with clear names
- Cross-references to related docs included