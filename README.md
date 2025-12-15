# FraudDetection

Quick start for running predict pipeline and avoiding ModuleNotFoundError

Options (choose one):

1) Run as module (recommended)

```powershell
# from project root
python -m src.pipeline.predict_pipeline
```

2) Use the provided wrapper which ensures the project root is in PYTHONPATH:

```powershell
# from project root
python run_predict.py
```

3) Install the package in editable mode (recommended for development):

```powershell
# from project root
pip install -e .
python -m src.pipeline.predict_pipeline
```

4) Set PYTHONPATH env var in powershell for a session:

```powershell
$env:PYTHONPATH = "$(Get-Location)"; python -m src.pipeline.predict_pipeline
```

Notes: If running via VS Code "Run File" or interactive script runner, use the wrapper or set the PYTHONPATH / run as module mode to preserve the package imports.
