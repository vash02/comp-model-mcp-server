# sandbox_executor.py (modified)

import subprocess
import sys
import json
import tempfile
import shutil
import os
from config import SANDBOX_TIMEOUT
from utils.logger import setup_logging

logger = setup_logging("sandbox_executor")

def run_in_sandbox(code_path: str, params: dict, use_temp_cwd: bool = False) -> dict:
    """
    Run the Python module at code_path in a separate process.
    If use_temp_cwd=True, creates a temporary directory, copies the code file there,
    and runs the subprocess with cwd=tempdir.
    """
    params_json = json.dumps(params)
    cmd = [sys.executable, os.path.basename(code_path), "--params", params_json]

    if use_temp_cwd:
        with tempfile.TemporaryDirectory() as workdir:
            # Copy code file into workdir
            dest = os.path.join(workdir, os.path.basename(code_path))
            shutil.copy(code_path, dest)
            logger.info(f"Running in sandbox cwd={workdir}: {cmd}")
            try:
                completed = subprocess.run(
                    cmd,
                    cwd=workdir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=SANDBOX_TIMEOUT,
                    text=True
                )
            except subprocess.TimeoutExpired:
                logger.error(f"Sandbox execution timed out after {SANDBOX_TIMEOUT}s for params {params}")
                return {"error": "timeout"}
    else:
        # Run in current working directory
        logger.info(f"Running sandbox: {cmd}")
        try:
            completed = subprocess.run(
                [sys.executable, code_path, "--params", params_json],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=SANDBOX_TIMEOUT,
                text=True
            )
        except subprocess.TimeoutExpired:
            logger.error(f"Sandbox execution timed out after {SANDBOX_TIMEOUT}s for params {params}")
            return {"error": "timeout"}

    # After subprocess completes:
    if completed.returncode != 0:
        logger.error(f"Sandbox execution failed (code {completed.returncode}). stderr:\n{completed.stderr.strip()}")
        return {"error": "execution_failed", "stderr": completed.stderr.strip()}

    stdout = completed.stdout.strip()
    if not stdout:
        logger.error("No stdout from subprocess; cannot parse JSON.")
        return {"error": "no_output"}

    try:
        last_line = stdout.splitlines()[-1]
        result = json.loads(last_line)
        return result
    except Exception as e:
        logger.error(f"Failed to parse JSON from stdout. stdout:\n{stdout}\nError: {e}")
        return {"error": "json_parse_failed", "stdout": stdout}
