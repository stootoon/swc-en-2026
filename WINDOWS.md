# Setup on Windows

The module READMEs give macOS/Linux commands. On Windows the same steps look a
little different: there is no `python3` command, the virtual environment keeps its
programs in `.venv\Scripts\` rather than `.venv/bin/`, and PowerShell blocks the
`activate` script by default. The instructions below avoid all three problems by
calling the environment's own `python` directly, so nothing needs activating.

## 1. Install Python and Git (once)

1. **Python**: download the installer from https://www.python.org/downloads/windows/
   (any 3.9 or newer). On the first screen tick **"Add python.exe to PATH"**, then
   choose *Install Now*. Avoid the Microsoft Store version: it sometimes fails to
   create virtual environments.
2. **Git**: download from https://git-scm.com/download/win and accept the defaults.

Close and reopen any terminal after installing. Then open **PowerShell** (press
the Windows key, type `powershell`, Enter) and check both work:

```powershell
python --version
git --version
```

If `python --version` opens the Microsoft Store instead of printing a version, go
to *Settings → Apps → Advanced app settings → App execution aliases* and switch off
the two "python" entries, then reopen PowerShell.

## 2. Get the course material (once)

```powershell
cd $HOME\Documents
git clone https://github.com/stootoon/swc-en-2026.git
```

## 3. Set up a module

Every module uses the same recipe. Pick the module from this table:

| module | folder | kernel name | display name |
|---|---|---|---|
| Behaviour Analysis | `behaviour` | `swc-behaviour` | `SWC Behaviour (.venv)` |
| Population Electrophysiology | `ephys-pop` | `swc-ephys-pop` | `SWC Ephys-Pop (.venv)` |
| Single Cell Electrophysiology | `ephys-one` | `swc-ephys-one` | `SWC Ephys-One (.venv)` |

Then run the following, replacing the folder, kernel name and display name with the
row you want. This example is for `ephys-one`:

```powershell
cd $HOME\Documents\swc-en-2026\ephys-one
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip setuptools wheel
.venv\Scripts\python -m pip install -e .
.venv\Scripts\python -m pip install jupyterlab ipykernel
.venv\Scripts\python -m ipykernel install --user --name swc-ephys-one --display-name "SWC Ephys-One (.venv)"
```

The `pip install -e .` step installs the module's backend package plus numpy, scipy
and matplotlib. It can take a few minutes the first time.

## 4. Start JupyterLab

```powershell
cd $HOME\Documents\swc-en-2026\ephys-one
.venv\Scripts\jupyter lab
```

A browser tab opens. Navigate into `notebooks/`, open a notebook, and pick the
module's kernel from the top-right corner of the notebook (for example
**"SWC Ephys-One (.venv)"**). To stop the server, press `Ctrl+C` in PowerShell.

## Troubleshooting

- **"running scripts is disabled on this system"**: you ran
  `.venv\Scripts\activate`. You don't need to. Use the `.venv\Scripts\python`
  form above instead, or allow scripts once with
  `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.
- **`python` is not recognised**: Python is not on your PATH. Rerun the installer,
  choose *Modify*, and tick *Add Python to environment variables*; or use the
  `py` launcher instead of `python` in the `python -m venv .venv` line.
- **The kernel is missing from the list in JupyterLab**: rerun the
  `ipykernel install` line, then refresh the browser tab.
- **`pip install -e .` fails with a build error**: make sure the pip upgrade line
  ran without error first, then try again.
- **Using Command Prompt (`cmd`) instead of PowerShell**: all the commands above
  work unchanged, except use `%USERPROFILE%\Documents` in place of
  `$HOME\Documents`.
