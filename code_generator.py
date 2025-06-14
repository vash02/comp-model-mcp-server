# code_generator.py

import os
import hashlib
import json
from openai import OpenAI
import re

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Please set the OPENAI_API_KEY environment variable")

client = OpenAI(api_key=OPENAI_API_KEY)
from typing import List, Dict
from config import OPENAI_MODEL, OPENAI_API_KEY, GENERATED_MODELS_DIR
from utils.logger import setup_logging

logger = setup_logging("code_generator")

if OPENAI_API_KEY is None:
    logger.error("OPENAI_API_KEY not set. Please export it in your environment.")

def strip_code_fences(code_text: str) -> str:
    """Remove Markdown-style triple backticks and language hints."""
    return re.sub(r"^```(?:python)?\n?|\n?```$", "", code_text.strip(), flags=re.MULTILINE)

def compute_model_id(
    model_name: str,
    equations: List[str],
    parameters: List[Dict[str, str]],
    initial_conditions: List[Dict[str, str]]
) -> str:
    """
    Deterministic model_id based on model_name, equations, parameters, and initial_conditions.
    """
    spec = {
        "model_name": model_name,
        "equations": equations,
        "parameters": parameters,
        "initial_conditions": initial_conditions
    }
    spec_json = json.dumps(spec, sort_keys=True)
    h = hashlib.md5(spec_json.encode()).hexdigest()[:8]
    return f"{model_name}_{h}"


def generate_code_from_spec(
    model_name: str,
    equations: List[str],
    parameters: List[Dict[str, str]],
    initial_conditions: List[Dict[str, str]]
) -> str:
    """
    Build a prompt for a physics model defined by multiple equations and JSON-spec parameters
    + initial conditions. Returns the generated Python code as a string.
    """
    # Format equations list
    eq_lines = "\n".join(f"{i+1}. {eq}" for i, eq in enumerate(equations))
    # Format parameter descriptions
    param_lines = "\n".join(f"- `{p['name']}`: {p['description']}" for p in parameters)
    # Format initial-condition descriptions
    ic_lines = ""
    if initial_conditions:
        ic_lines = "\nInitial Conditions (JSON object under key 'initial_conditions'):\n" + \
                   "\n".join(f"- `{ic['name']}`: {ic['description']}" for ic in initial_conditions)

    prompt = f"""
You are an expert Python developer experienced in physics simulations.
Generate a Python module that implements the following physics model:

Equations (LaTeX or Pythonic):
{eq_lines}

Parameter descriptions (JSON under key 'params'):
{param_lines}{ic_lines}

Requirements:
1. The code must parse a single JSON input from a command-line argument `--params`.
   That JSON must include:
     • A mapping `params` of parameter names to values.
     • A mapping `initial_conditions` of initial-condition names to values.
2. Use SymPy when helpful to derive any necessary system equations.
3. Include imports: sympy, numpy, scipy.integrate, json, sys, argparse, etc.
4. Define `run_experiment(input_dict: dict) -> dict` that:
     • Reads `params = input_dict["params"]` and `ics = input_dict["initial_conditions"]`.
     • Converts all values to floats.
     • Computes any analytic expressions if needed.
     • Uses SciPy’s `solve_ivp` (or suitable integrator) with the provided initial conditions.
     • Returns a JSON-serializable dict of results (floats, lists).

5. In the `if __name__ == "__main__":` block:
     • Parse `--params` (JSON string) into `input_dict`.
     • Call `run_experiment(input_dict)` and `print(json.dumps(results))`.

Ensure no SymPy objects remain in the output. Return only the complete Python code (no extra explanation).
""".strip()

    logger.info("Sending prompt to OpenAI for code generation...")
    response = client.chat.completions.create(model=OPENAI_MODEL,
    messages=[{"role": "user", "content": prompt}],
    max_tokens=1500,
    temperature=0.0)
    raw_code = response.choices[0].message.content
    code = strip_code_fences(raw_code)
    if "def run_experiment" not in code:
        logger.warning("Generated code missing `run_experiment`. Please inspect manually.")
    return code


def save_generated_code(model_id: str, code: str) -> str:
    """
    Save the generated Python module under GENERATED_MODELS_DIR/<model_id>.py.
    Returns the full file path.
    """
    os.makedirs(GENERATED_MODELS_DIR, exist_ok=True)
    filepath = os.path.join(GENERATED_MODELS_DIR, f"{model_id}.py")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(code)
    logger.info(f"Saved generated code to {filepath}")
    return filepath
