# Minerva RL Documentation

This site is built for GitHub Pages and includes:

- the latest generated run report from `reports/latest/README.md`
- source snapshots for everything in `hyperparameters/`
- auto-generated API reference pages for all Python modules in `trainers/`
- math-annotated trainer pages that render inline `# LaTeX:` comments

## Quick Links

- [Latest Report](generated/report/latest.md)
- [Hyperparameters](generated/hyperparameters/index.md)
- [Trainers API](generated/trainers-api/index.md)
- [Trainers Math](generated/trainers-math/index.md)

## Build Process

Docs content under `docs/generated/` is re-created by:

```bash
python scripts/generate_docs_content.py
```

GitHub Pages runs this step automatically in CI before `mkdocs build`.
