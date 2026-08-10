"""Ensures the repo root is on sys.path so `import pair_crew...` works
under plain `pytest` (not just `python -m pytest`) -- there's no
pyproject.toml/setup.py installing this project as a package, so pytest's
own rootdir insertion is what makes the import resolve."""
