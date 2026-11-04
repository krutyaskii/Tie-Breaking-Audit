"""Build a self-contained Colab notebook without publishing project changes."""

import base64
import io
import json
import textwrap
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[1]
paths = [ROOT / name for name in (
    "README.md", "prepare_data.py", "requirements-rotate.txt", "reports/data_manifest.json")]
for directory in ("rotate", "vendor", "configs", "tests", "docs"):
    paths.extend(path for path in (ROOT / directory).rglob("*")
                 if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc")
archive = io.BytesIO()
with ZipFile(archive, "w", compression=ZIP_DEFLATED) as stream:
    for path in sorted(paths):
        entry = ZipInfo(str(path.relative_to(ROOT)).replace("\\", "/"), date_time=(1980, 1, 1, 0, 0, 0))
        entry.external_attr = 0o644 << 16
        stream.writestr(entry, path.read_bytes(), compress_type=ZIP_DEFLATED)
payload = base64.b64encode(archive.getvalue()).decode("ascii")
cells = []


def markdown(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(text).strip() + "\n"})


def code(text):
    source = textwrap.dedent(text).strip() + "\n"
    compile(source, "<notebook-cell>", "exec")
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source})


markdown("""
    # RotatE: training and evaluator handoff

    **Before running:** select **Runtime > Change runtime type > T4 GPU**.
    Then run the cells in order. This notebook includes the source code, so no
    GitHub push is required. It downloads the pinned FB15k-237 archive itself.

    The default `colab_budget` preset trains a smaller, documented checkpoint.
    It is not the paper's exact hyperparameter setting. An optional `paper`
    preset is available and needs considerably more GPU time. Results apply to
    the checkpoint actually trained. The four-step smoke run is only a test.

    Real training has not happened merely because this notebook opens. It is
    complete only after the training cell finishes and `training_complete` is
    true. The last cells create and download the evaluator package.
    """)
code("""
    import base64, io, os, zipfile
    from pathlib import Path
    PROJECT = Path('/content/tie_breaking_audit')
    PROJECT.mkdir(parents=True, exist_ok=True)
    SOURCE_ARCHIVE = base64.b64decode(PAYLOAD_PLACEHOLDER)
    with zipfile.ZipFile(io.BytesIO(SOURCE_ARCHIVE)) as source:
        for name in source.namelist():
            destination = PROJECT / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source.read(name))
    os.chdir(PROJECT)
    print('Source restored:', PROJECT)
    """.replace("PAYLOAD_PLACEHOLDER", repr(payload)))
markdown("""
    ## 1. Check the GPU and dependencies

    Use Colab's existing CUDA-enabled PyTorch. Do not install a CPU-only wheel.
    Exact installed versions will be recorded with the trained checkpoint.
    """)
code("""
    import importlib.util, subprocess, sys
    import os, queue, signal, threading, time

    def run_live(command, heartbeat_seconds=30):
        print('Running:', ' '.join(map(str, command)), flush=True)
        started = time.monotonic()
        child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, bufsize=1,
                                 env={**os.environ, 'PYTHONUNBUFFERED': '1'})
        output = queue.Queue()
        def collect_output():
            try:
                for line in child.stdout:
                    output.put(line)
            finally:
                output.put(None)
        threading.Thread(target=collect_output, daemon=True).start()
        try:
            while True:
                try:
                    line = output.get(timeout=heartbeat_seconds)
                except queue.Empty:
                    elapsed = time.monotonic() - started
                    print(f'[status] Process {child.pid} is still active; elapsed {elapsed / 60:.1f} min. '
                          'No new child output. This is a liveness check, not a training-progress measurement.',
                          flush=True)
                    continue
                if line is None:
                    break
                print(line, end='', flush=True)
            returncode = child.wait()
            if returncode:
                raise subprocess.CalledProcessError(returncode, command)
        except KeyboardInterrupt:
            if child.poll() is None:
                child.send_signal(signal.SIGINT)
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    print(f'Process {child.pid} has not stopped yet. Do not start a second training process.',
                          flush=True)
            raise
        finally:
            if child.poll() is not None:
                child.stdout.close()

    if importlib.util.find_spec('psutil') is None:
        run_live([sys.executable, '-m', 'pip', 'install', 'psutil>=5.9'])
    import torch, numpy as np
    assert torch.cuda.is_available(), 'Select Runtime > Change runtime type > T4 GPU, then reconnect.'
    print('GPU:', torch.cuda.get_device_name(0))
    print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda, 'NumPy:', np.__version__)
    """)
markdown("""
    ## 2. Prepare data and run correctness checks

    This preserves original splits and IDs, verifies the archive and file hashes,
    and creates dictionaries for both model owners. Tests cover the score
    direction, head/tail candidate IDs, mapping protection, known-answer filtering,
    expected random MRR, sampler resume, and loss/update agreement with upstream.
    """)
code("""
    run_live([sys.executable, 'prepare_data.py'])
    run_live([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'])
    """)
markdown("""
    ## 3. Run a short smoke test

    These tiny weights only check that saving, resuming, and scoring work.
    They must not be reported as the project result.
    """)
code("""
    smoke_dir = Path('/content/rotate_smoke')
    command = [sys.executable, '-u', '-m', 'rotate.train', '--config', 'configs/rotate_smoke.json',
               '--output', str(smoke_dir), '--device', 'cuda']
    if (smoke_dir / 'latest.pt').exists():
        command.append('--resume')
    run_live(command)
    run_live([sys.executable, '-m', 'rotate.verify', '--checkpoint', str(smoke_dir / 'best.pt'),
              '--device', 'cuda'])
    """)
markdown("""
    ## 4. Choose the real configuration and save location

    The recommended `colab_budget` preset uses dimension 200 and 30,000 updates.
    `paper` uses dimension 1000 and 150,000 updates from the upstream README.
    See the full comparison in `docs/ROTATE.md`.

    Google Drive is recommended for resumable checkpoints. You will need to
    approve the normal Drive sign-in/access prompt yourself. Setting
    `USE_GOOGLE_DRIVE = False` keeps everything in the temporary runtime; download
    outputs before disconnecting. No paid compute is required or purchased.
    """)
code("""
    PRESET = 'colab_budget'  # Change to 'paper' only if you have enough GPU time.
    RUN_TAG = 'seed42'       # Use a new tag if you change the model/configuration.
    USE_GOOGLE_DRIVE = True
    assert PRESET in ('colab_budget', 'paper')
    if USE_GOOGLE_DRIVE:
        from google.colab import drive
        drive.mount('/content/drive')
        SAVE_ROOT = Path('/content/drive/MyDrive/TieBreakingAudit')
    else:
        SAVE_ROOT = Path('/content/TieBreakingAudit')
    RUN_DIR = SAVE_ROOT / f'rotate_{PRESET}_{RUN_TAG}'
    CONFIG = PROJECT / 'configs' / f'rotate_{PRESET}.json'
    SAVE_ROOT.mkdir(parents=True, exist_ok=True)
    source_backup = SAVE_ROOT / 'rotate_source_bundle.zip'
    if source_backup.exists() and source_backup.read_bytes() != SOURCE_ARCHIVE:
        raise RuntimeError('A different source bundle already exists. Use a new SAVE_ROOT for changed code.')
    source_backup.write_bytes(SOURCE_ARCHIVE)
    print('Configuration:', CONFIG.read_text())
    print('Training output:', RUN_DIR)
    """)
markdown("""
    ## 5. Train (or resume) the real model

    This is the long-running cell. Every 100 updates, it prints loss, measured
    seconds per update, and an estimate excluding validation/export overhead.
    Full validation selects the checkpoint. Final test queries are not scored.
    A status line appears every 30 seconds if the child process has not produced
    output. Initial full validation happens before the first optimizer update.

    If interrupted, reconnect the same GPU setup, rerun the preceding cells,
    keep the same preset/tag, and rerun this cell. It resumes from the latest
    saved checkpoint, normally at most 1,000 updates behind. Free Colab may
    disconnect; Drive preserves saved checkpoints but does not keep GPU compute
    running after the runtime has ended.
    """)
code("""
    command = [sys.executable, '-u', '-m', 'rotate.train', '--config', str(CONFIG),
               '--output', str(RUN_DIR), '--device', 'cuda']
    if (RUN_DIR / 'latest.pt').exists():
        command.append('--resume')
    run_live(command)
    import json
    summary = json.loads((RUN_DIR / 'summary.json').read_text())
    print(json.dumps(summary, indent=2))
    assert summary['training_complete'], 'Training is incomplete; resume before a final export.'
    """)
markdown("""
    ## 6. Verify and build the evaluator package

    Export includes the selected weights, dictionaries, exact environment,
    configuration, training/validation logs, source attribution, adapter checks,
    and a small validation score example. Dataset splits and optimizer state
    are excluded. `README.md` inside the bundle explains the scoring interface.
    """)
code("""
    from datetime import datetime, timezone
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    HANDOFF_DIR = SAVE_ROOT / f'rotate_{PRESET}_handoff_{stamp}'
    run_live([sys.executable, '-u', '-m', 'rotate.export', '--run', str(RUN_DIR),
              '--output', str(HANDOFF_DIR), '--device', 'cuda'])
    HANDOFF_ZIP = HANDOFF_DIR.with_suffix('.zip')
    print('Evaluator package:', HANDOFF_ZIP)
    print('Archive checksum:', HANDOFF_ZIP.with_suffix('.zip.sha256').read_text())
    """)
markdown("""
    ## 7. Download and hand off

    Send the ZIP and its `.zip.sha256` file to your evaluator. Keep the Drive run
    folder for provenance and resuming. Share the checkpoint's actual preset and
    selected step. The evaluator should calculate all tie policies from one set
    of raw scores on one recorded device. This notebook does not push to GitHub.
    """)
code("""
    from google.colab import files
    files.download(str(HANDOFF_ZIP))
    files.download(str(HANDOFF_ZIP.with_suffix('.zip.sha256')))
    """)
for index, cell in enumerate(cells):
    cell["metadata"]["id"] = f"rotate_cell_{index:02d}"
notebook = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {"colab": {"provenance": []}, "accelerator": "GPU",
                 "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
    "cells": cells,
}
target = ROOT / "notebooks/RotatE_Colab.ipynb"
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps(notebook, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
print(f"Notebook ready: {target}")
print(f"Embedded source: {len(archive.getvalue()):,} bytes; notebook: {target.stat().st_size:,} bytes")
