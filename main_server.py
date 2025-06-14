# main_server.py

import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from utils.logger import setup_logging
from datastore import DataStore
from code_generator import compute_model_id, generate_code_from_spec, save_generated_code
from sandbox_executor import run_in_sandbox
from config import GENERATED_MODELS_DIR

logger = setup_logging("main_server")
# Singleton datastore
datastore = DataStore()
app = FastAPI(
    title="MCP Server for Physics Models",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    swagger_ui_parameters={
            "syntaxHighlight": {
                "activate": True,     # must be true
                "theme": "github"     # or "monokai", "arta", etc.
            }
        }
)


@app.on_event("startup")
def ensure_tables_exist():
    # Call DataStore's internal init, which uses CREATE TABLE IF NOT EXISTS
    datastore._init_tables()

# --- Pydantic models ---
class ParamValue(BaseModel):
    name: str
    value: Any

class ParamDesc(BaseModel):
    name: str
    description: str
    value: str

class CreateModelRequest(BaseModel):
    model_name: str
    equations: List[str]                   # now accepts multiple equations
    parameters: List[ParamDesc]
    initial_conditions: Optional[List[ParamDesc]] = []  # describe any ICs

class CreateModelResponse(BaseModel):
    model_id: str
    code: str

class ModelCode(BaseModel):
    model_id: str
    code: str

class RunExperimentRequest(BaseModel):
    parameters: List[ParamValue]
    initial_conditions: Optional[List[ParamValue]] = []

class RunExperimentResponse(BaseModel):
    parameters: List[ParamValue]
    initial_conditions: List[ParamValue]
    results: Dict[str, Any]

class BatchRunRequest(BaseModel):
    runs: List[RunExperimentRequest]

class ExperimentRecord(BaseModel):
    id: int
    params: Dict[str, Any]
    initial_conditions: Dict[str, Any]
    results: Dict[str, Any]
    created_at: str

# --- Endpoints ---

@app.get("/", tags=["Health"])
def read_root():
    return {"status": "MCP Server is running", "models_count": datastore.count_models()}

@app.post("/models", response_model=CreateModelResponse)
def create_model(req: CreateModelRequest):
    model_name = req.model_name.strip()
    equations = [eq.strip() for eq in req.equations]
    parameters = [{"name": p.name, "description": p.description, "value": p.value} for p in req.parameters]
    ics = [{"name": ic.name, "description": ic.description, "value": ic.value} for ic in req.initial_conditions]
    model_id = compute_model_id(model_name, equations, parameters, ics)

    if datastore.model_exists(model_id):
        rec = datastore.get_model(model_id)
        with open(rec["code_path"], "r", encoding="utf-8") as f:
            code_text = f.read()
        return {"model_id": model_id, "code": code_text}

    # Generate and save code
    try:
        code = generate_code_from_spec(model_name, equations, parameters, ics)
    except Exception as e:
        logger.error(f"Error generating code: {e}")
        raise HTTPException(status_code=500, detail="Error generating model code")

    code_path = save_generated_code(model_id, code)
    spec = {
        "model_name": model_name,
        "equations": equations,
        "parameters": parameters,
        "initial_conditions": ics
    }
    datastore.store_model(model_id, model_name, spec, code_path)
    datastore.set_approved(model_id, False)

    return {"model_id": model_id, "code": code}


@app.get("/models/{model_id}", response_model=Dict[str, Any])
def get_model(model_id: str):
    rec = datastore.get_model(model_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Model not found")
    return rec


# at the top of main_server.py, add:
from fastapi.responses import PlainTextResponse

# … then replace the existing get_model_code endpoint with:

@app.get(
    "/models/{model_id}/code",
    response_class=PlainTextResponse,
    summary="Fetch the generated Python module (plain text)"
)
def get_model_code(model_id: str):
    rec = datastore.get_model(model_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Model not found")
    code_path = rec["code_path"]
    try:
        with open(code_path, "r", encoding="utf-8") as f:
            code = f.read()
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Generated code file missing")
    return code



@app.post("/models/{model_id}/approve")
def approve_model(model_id: str):
    if not datastore.model_exists(model_id):
        raise HTTPException(status_code=404, detail="Model not found")
    datastore.set_approved(model_id, True)
    return {"detail": f"Model {model_id} approved for execution."}


@app.post("/models/{model_id}/run", response_model=RunExperimentResponse)
def run_experiment(model_id: str, req: RunExperimentRequest):
    # 1) Model existence & approval
    if not datastore.model_exists(model_id):
        raise HTTPException(404, "Model not found")
    if not datastore.is_approved(model_id):
        raise HTTPException(
            403,
            "Model not approved. Review via /models/{model_id}/code and POST to /models/{model_id}/approve."
        )

    # 2) Build the nested payload
    params_dict = {p.name: p.value for p in req.parameters}
    ics_dict    = {ic.name: ic.value for ic in (req.initial_conditions or [])}
    payload = {
        "params": params_dict,
        "initial_conditions": ics_dict
    }

    # 3) Check cache
    existing = datastore.find_experiment(model_id, payload)
    if existing:
        # reconstruct ParamValue lists
        prev_params = [ParamValue(name=k, value=v) for k, v in existing["params"].items()]
        prev_ics    = [ParamValue(name=k, value=v) for k, v in existing["initial_conditions"].items()]
        return RunExperimentResponse(
            parameters=prev_params,
            initial_conditions=prev_ics,
            results=existing["results"]
        )

    # 4) Execute in sandbox
    model_meta = datastore.get_model(model_id)
    results = run_in_sandbox(model_meta["code_path"], payload)

    # 5) Store experiment
    datastore.store_experiment(model_id, payload["params"], payload["initial_conditions"], results)

    # 6) Return
    return RunExperimentResponse(
        parameters=req.parameters,
        initial_conditions=req.initial_conditions or [],
        results=results
    )

@app.post("/models/{model_id}/batch_run", response_model=List[RunExperimentResponse])
def batch_run(model_id: str, req: BatchRunRequest):
    if not datastore.model_exists(model_id):
        raise HTTPException(status_code=404, detail="Model not found")
    if not datastore.is_approved(model_id):
        raise HTTPException(
            status_code=403,
            detail="Model code not approved. Review via /models/{model_id}/code and POST to /models/{model_id}/approve."
        )

    responses: List[RunExperimentResponse] = []
    model_meta = datastore.get_model(model_id)
    code_path = model_meta["code_path"]

    for run in req.runs:
        all_inputs = {**run.params, **run.initial_conditions}
        existing = datastore.find_experiment(model_id, all_inputs)
        if existing:
            resp = {
                "params": existing["params"],
                "initial_conditions": existing["initial_conditions"],
                "results": existing["results"]
            }
        else:
            results = run_in_sandbox(code_path, all_inputs)
            datastore.store_experiment(model_id, all_inputs, results)
            resp = {
                "params": run.params,
                "initial_conditions": run.initial_conditions or {},
                "results": results
            }
        responses.append(resp)

    return responses


@app.get("/models/{model_id}/results", response_model=List[ExperimentRecord])
def get_results(model_id: str):
    if not datastore.model_exists(model_id):
        raise HTTPException(status_code=404, detail="Model not found")
    records = datastore.get_experiments(model_id)
    return [
        ExperimentRecord(
            id=rec["id"],
            params={k: rec["params"][k] for k in rec["params"] if k in rec["params"]},
            initial_conditions={k: rec["params"][k] for k in rec["initial_conditions"]} if "initial_conditions" in rec else {},
            results=rec["results"],
            created_at=rec["created_at"]
        )
        for rec in records
    ]


@app.delete("/models/{model_id}")
def delete_model(model_id: str):
    if not datastore.model_exists(model_id):
        raise HTTPException(status_code=404, detail="Model not found")

    conn = datastore.conn
    cur = conn.cursor()
    cur.execute("DELETE FROM experiments WHERE model_id = ?", (model_id,))
    cur.execute("DELETE FROM models WHERE model_id = ?", (model_id,))
    conn.commit()

    code_file = os.path.join(GENERATED_MODELS_DIR, f"{model_id}.py")
    if os.path.isfile(code_file):
        try:
            os.remove(code_file)
            logger.info(f"Deleted code file {code_file}")
        except Exception as e:
            logger.warning(f"Failed to delete code file: {e}")

    return {"detail": f"Deleted model {model_id} and its experiments."}
