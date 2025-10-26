# Git Workflow and Best Practices

This document defines the repository’s Git workflow, version compatibility guidance, and recommended practices. It’s designed to keep history clean, reviews efficient, and automation predictable.

## Version Compatibility
- Recommended: `git >= 2.44`
- macOS: `brew install git` or use Xcode’s Git if recent
- Windows: Install the latest Git for Windows
- Linux: Use distro packages or `ppa:git-core/ppa` for up-to-date versions

## Branching Strategy
- Default branch: `main` (protected)
- Use short-lived feature branches: `feature/<slug>`, `fix/<slug>`, `docs/<slug>`
- Avoid long-running personal branches; rebase regularly to reduce drift

## Commit Conventions (Conventional Commits)
- Format: `type(scope): short summary`
- Types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `chore`, `build`, `ci`
- Examples:
  - `feat(experiments): add DEIM size-aware prompts`
  - `docs(readme): update repository structure`
  - `refactor(deim): lazy-load heavy imports for fast --help`

## Pull Requests
- Target: `main`
- Keep PRs small and focused (≤ 300 lines changed when possible)
- Include:
  - Clear description, motivation, and scope
  - Links to related issues/experiments
  - Test results (e.g., `pytest -q` output)
  - Changelog updates when user-facing changes occur
- Prefer rebase to keep linear history, but merge commits are acceptable if the PR is large and complex

## Reviews and Approvals
- At least one reviewer approval required for code changes
- For documentation-only changes: optional approval if small
- Blockers: failing tests, lint errors, or unresolved TODOs

## Tags and Releases
- Tag format: `vX.Y.Z` (semantic versioning)
- Update `CHANGELOG.md` with notable changes
- Use GitHub Releases for packaging artifacts if needed

## Recommended Settings
- Sign commits if possible: `git config --global commit.gpgsign true`
- Use sparse checkout for large repos: `git sparse-checkout set <paths>`
- Use partial clone when bandwidth constrained: `git clone --filter=blob:none <url>`
- Install pre-commit hooks to enforce formatting/tests:
  - Add `.pre-commit-config.yaml` and run `pre-commit install`

## Common Commands
- Initialize local tracking:
  ```bash
  git init
  git remote add origin <repo_url>
  git fetch origin
  git switch -c feature/<slug>
  ```
- Sync feature branch with `main`:
  ```bash
  git fetch origin
  git rebase origin/main
  # resolve conflicts, then
  git push --force-with-lease
  ```
- Create a pull request (CLI):
  ```bash
  gh pr create --fill --base main --head feature/<slug>
  ```
- Tag a release:
  ```bash
  git tag -a v1.0.0 -m "Initial consolidated release"
  git push origin v1.0.0
  ```

## Notes
- Submodules should be avoided unless necessary; prefer vendoring small utilities
- Large binary datasets should use Git LFS or DVC, not regular Git
- Keep `main` green: tests must pass before merging