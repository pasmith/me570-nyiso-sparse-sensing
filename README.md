# ME 570: NYISO Sparse Sensing

## Prerequisites

- [uv](https://astral.sh) installed (`curl -LsSf https://astral.sh | sh`)
- VS Code with the **Python** and **Jupyter** extensions installed

## Setup

1. **Install dependencies and create the virtual environment:**
   ```bash
   uv sync
   ```

2. **Add `src/` to the virtual environment's Python path:**
   Run the following from the project root so notebooks and scripts can import modules directly from `src/` (e.g., `from data_loader import load_data`):
   ```bash
   echo "$(pwd)/src" > .venv/lib/python3.15/site-packages/local_src.pth
   ```
   *(Alternatively, resolve the path dynamically regardless of Python minor version:)*
   ```bash
   echo "$(pwd)/src" > "$(.venv/bin/python -c 'import site; print(site.getsitepackages())')/local_src.pth"
   ```

## Running Notebooks in VS Code

1. Open the repository root folder (`me570-nyiso-sparse-sensing/`) in VS Code.
2. Ensure `.vscode/settings.json` includes:
   ```json
   {
       "python.analysis.extraPaths": [
           "${workspaceFolder}/src"
       ],
       "jupyter.notebookFileRoot": "${workspaceFolder}/notebooks"
   }
   ```
3. Open any notebook under `notebooks/`.
4. Click **Select Kernel** (top right) -> **Python Environments** -> select `.venv` (`Python 3.15.2`).
5. Click **Restart** on the notebook toolbar before running cells.

## Project Structure

```text
me570-nyiso-sparse-sensing/
├── .vscode/
│   └── settings.json      # VS Code & Pylance path configuration
├── notebooks/             # Jupyter notebooks for analysis & experiments
├── src/                   # Core Python modules (e.g., data_loader.py)
├── pyproject.toml         # Project dependencies managed by uv
└── README.md
```
