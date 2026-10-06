# Releasing to PyPI

1. Bump `version` in `pyproject.toml`. PyPI won't accept a version twice.
2. Build from a clean `dist/`, since `uv publish` uploads everything in it:

   ```sh
   Remove-Item -Recurse -Force dist
   uv build
   ```

3. Optional: run the built wheel to make sure it works (icons included):

   ```sh
   uv run --isolated --no-project --with dist/pyrpoc-X.Y.Z-py3-none-any.whl pyrpoc
   ```

4. Publish, then close the terminal so the token isn't left in it:

   ```sh
   uv publish --token pypi-...
   ```

5. Check it installs: `uv tool upgrade pyrpoc`.
