# Releasing to PyPI

How to publish a new version of `pyrpoc` so that `uv tool install pyrpoc` and
`uv tool upgrade pyrpoc` pick it up. Everything goes through `uv`; no `twine`
or `build` install is needed.

## 1. Prepare the release on a branch

1. Make sure `main` has everything that should ship, then branch off it
   (for example `release/vX.Y.Z`).
2. Bump `version` in `pyproject.toml`. It is the only place the version lives.
   PyPI never accepts the same version twice, even after a deletion, so pick a
   new number every time.
3. Run the checks and fix anything they report:

   ```sh
   uv sync
   uv run pre-commit run --all-files --config .github/pre-commit.yaml
   ```

4. Launch the app once with `uv run pyrpoc` to check that it starts.
5. Commit, open a PR, and merge it into `main`. Publish from the merged
   `main`, so the version on PyPI matches a commit.

## 2. Build

```sh
git switch main
git pull
rm -rf dist build            # PowerShell: Remove-Item -Recurse -Force dist, build
uv build
```

Clearing `dist/` matters: `uv publish` uploads everything in it, and old wheels
from earlier releases sit there otherwise.

`uv build` writes `dist/pyrpoc-X.Y.Z.tar.gz` and
`dist/pyrpoc-X.Y.Z-py3-none-any.whl`.

## 3. Check the build before uploading

Install the wheel into a throwaway environment and run it. This catches missing
files, such as the icons in `pyrpoc/assets/`, which ship only because of
`[tool.setuptools.package-data]` in `pyproject.toml`.

```sh
uv run --isolated --no-project --with dist/pyrpoc-X.Y.Z-py3-none-any.whl pyrpoc
```

To list what went into the wheel without installing it:

```sh
python -m zipfile -l dist/pyrpoc-X.Y.Z-py3-none-any.whl
```

## 4. Publish

`uv publish` reads the token from the `UV_PUBLISH_TOKEN` environment variable,
so it never has to appear in the command or shell history.

```sh
# bash
export UV_PUBLISH_TOKEN=pypi-...
uv publish
```

```powershell
# PowerShell
$env:UV_PUBLISH_TOKEN = "pypi-..."
uv publish
```

`uv publish --token pypi-...` also works but leaves the token in history.

### Optional: dry run on TestPyPI

TestPyPI needs its own account and token, separate from PyPI's.

```sh
UV_PUBLISH_TOKEN=<testpypi token> uv publish --publish-url https://test.pypi.org/legacy/
```

## 5. After publishing

1. Tag the commit and push the tag:

   ```sh
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```

2. Optionally create a GitHub release from the tag:
   `gh release create vX.Y.Z dist/* --generate-notes`.
3. Check the install as a user would. PyPI can take a minute to serve the new
   version.

   ```sh
   uv tool upgrade pyrpoc     # or: uv tool install pyrpoc
   pyrpoc
   ```

4. Check the page at <https://pypi.org/project/pyrpoc/>. The long description
   there is `README.md`.

## If something goes wrong

- **"File already exists"**: that version is already on PyPI. Bump the
  version and rebuild. Uploads can't be overwritten.
- **403 / invalid credentials**: the token is wrong, expired, or scoped to a
  different project. The username is always `__token__`, and `uv` sets it for
  you when you use a token.
- **A broken release shipped**: *yank* it on PyPI (Manage project → Releases →
  Options → Yank). Yanking hides the release from new installs without breaking
  pinned ones. Then fix it and publish the next patch version.
