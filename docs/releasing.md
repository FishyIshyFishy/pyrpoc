
1. Bump `version` in [pyproject.toml](pyproject.toml). 
2. Clear out `dist/` and build. `uv publish` uploads everything in `dist/`
   ```powershell
   Remove-Item -Recurse -Force dist
   uv build
   ```
3. Publish, then close the terminal
   ```powershell
   uv publish --token pypi-...
   ```
4. Check that it installs: `uv tool upgrade pyrpoc`