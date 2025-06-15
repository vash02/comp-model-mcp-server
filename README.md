# Physics Model MCP Server

A modular platform for defining, generating, executing, and tracking physics model experiments with reproducibility.

## 🚀 Quick Start

1. **Clone & install dependencies**
   ```bash
   git clone https://github.com/your-org/comp-model-mcp-server.git
   cd comp-model-mcp-server
   pip install -r requirements.txt
   ```

2. **Set environment variables**
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   export DB_PATH="mcp_server.db"
   export GENERATED_MODELS_DIR="generated_models"
   export SANDBOX_TIMEOUT=30
   ```

3. **Run the server**
   ```bash
   uvicorn main_server:app --reload
   ```

---

## 🔁 Reproduce an Experiment

### 1. Create a Model
Submit a model specification:
```bash
curl -X POST http://localhost:8000/models \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "lorenz",
    "equations": ["dx/dt = sigma*(y - x)", "dy/dt = x*(rho - z) - y", "dz/dt = x*y - beta*z"],
    "parameters": [
      {"name":"sigma","description":"Prandtl number","value":"10.0"},
      {"name":"rho","description":"Rayleigh number","value":"28.0"},
      {"name":"beta","description":"Geometric factor","value":"8.0/3.0"}
    ],
    "initial_conditions":[
      {"name":"x0","description":"initial x","value":"1.0"},
      {"name":"y0","description":"initial y","value":"1.0"},
      {"name":"z0","description":"initial z","value":"1.0"}
    ]
}'
```
Save the returned `model_id`.

### 2. Approve the Model
```bash
curl -X POST http://localhost:8000/models/{model_id}/approve
```

### 3. Run a Single Experiment
```bash
curl -X POST http://localhost:8000/models/{model_id}/run \
  -H "Content-Type: application/json" \
  -d '{
    "parameters":[{"name":"sigma","value":10.0},{"name":"rho","value":28.0},{"name":"beta","value":2.6667}],
    "initial_conditions":[{"name":"x0","value":1.0},{"name":"y0","value":1.0},{"name":"z0","value":1.0}]
}'
```
> Response includes experiment `results`, `duration`, and `exit_code`.

### 4. Retrieve Experiment Results
```bash
curl http://localhost:8000/models/{model_id}/results
```

---

## 🧬 CLI-Based Experiment Reproduction

1. **Fetch script**
   ```bash
   curl http://localhost:8000/models/{model_id}/code -o model.py
   ```

2. **Run the script locally**
   ```bash
   python model.py --params '{
     "params":{"sigma":10.0,"rho":28.0,"beta":2.6667},
     "initial_conditions":{"x0":1.0,"y0":1.0,"z0":1.0}
   }'
   ```
   The output will match the experiment results.

---

## 🛠️ Cleanup (Optional)

Delete the model and all associated experiments:
```bash
curl -X DELETE http://localhost:8000/models/{model_id}
```

---

## 🤖 Embeddings & Analytics

Stored models and experiments include:

- **Text embeddings** of model specification
- **Numeric + text joint embeddings** of experiment parameters/results

These can be used for clustering, similarity search, or advanced analytics.

---

## 📦 Requirements

- Python 3.10+
- Dependencies in `requirements.txt`:
  - `fastapi`, `uvicorn`, `openai`, `sentence-transformers`, `torch`, `numpy`, `scipy`, `sympy`, etc.

To ensure reproducibility, use the same environment setup and input parameters.