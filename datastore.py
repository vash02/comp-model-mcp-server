# datastore.py
import hashlib
import sqlite3
import json
import threading
from datetime import datetime
from config import DB_PATH
from utils.embedding_utils import embed_experiment
from utils.logger import setup_logging
from utils.embedding_utils import embed_model_spec


logger = setup_logging()

class DataStore:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, db_path=None):
        # Singleton pattern so that FastAPI reuse same connection object if desired
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(DataStore, cls).__new__(cls)
                    cls._instance._init(db_path or DB_PATH)
        return cls._instance

    def _init(self, db_path):
        self.db_path = db_path
        # For SQLite with FastAPI, allow check_same_thread=False
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_tables()

    def _init_tables(self):
        cur = self.conn.cursor()
        # models table
        cur.execute("""
        CREATE TABLE IF NOT EXISTS models (
          model_id TEXT PRIMARY KEY,
          model_name TEXT,
          spec TEXT,
          code_path TEXT,
          approved INTEGER DEFAULT 0,
          embedding TEXT,
          created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        """)
        # experiments table
        cur.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
          exp_id TEXT PRIMARY KEY,
          model_id TEXT,
          params TEXT,
          initial_conditions TEXT,
          vector TEXT,
          results TEXT,
          embedding TEXT,
          created_at DATETIME,
          FOREIGN KEY(model_id) REFERENCES models(model_id)
        )
        """)
        self.conn.commit()

    # Model metadata methods
    def model_exists(self, model_id: str) -> bool:
        cur = self.conn.cursor()
        cur.execute("SELECT 1 FROM models WHERE model_id = ?", (model_id,))
        return cur.fetchone() is not None

    def count_models(self) -> int:
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM models")
        return cur.fetchone()[0]

    def store_model(self, model_id: str, model_name: str, spec: dict, code_path: str):
        """
        Store model metadata and compute/store joint embedding for the model spec.
        """

        cur = self.conn.cursor()
        spec_json = json.dumps(spec)
        now = datetime.utcnow().isoformat()

        # Compute spec embedding
        try:
            model_emb = embed_model_spec(spec)
            emb_json = json.dumps(model_emb)
        except Exception as e:
            logger.warning(f"Failed to compute model embedding: {e}")
            emb_json = None

        cur.execute("""
        INSERT OR REPLACE INTO models
          (model_id, model_name, spec, code_path, embedding, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (
            model_id,
            model_name,
            spec_json,
            code_path,
            emb_json,
            now
        ))
        self.conn.commit()
        logger.info(f"Stored model metadata: {model_id}")

    def get_model(self, model_id: str):
        cur = self.conn.cursor()
        cur.execute("SELECT model_name, spec, code_path, created_at FROM models WHERE model_id = ?", (model_id,))
        row = cur.fetchone()
        if not row:
            return None
        return {
            "model_id": model_id,
            "model_name": row["model_name"],
            "spec": json.loads(row["spec"]),
            "code_path": row["code_path"],
            "created_at": row["created_at"]
        }

    # Experiment methods
    def store_experiment(
            self,
            model_id: str,
            params: dict,
            initial_conditions: dict,
            results: dict
    ):
        """
        Store a single experiment run, saving:
          - params & initial_conditions & results (as JSON)
          - normalized numeric vector (JSON list)
          - joint text+numeric embedding (JSON list)
        """

        cur = self.conn.cursor()
        now = datetime.utcnow().isoformat()

        # JSON‐encode inputs
        params_json = json.dumps(params, sort_keys=True)
        ics_json = json.dumps(initial_conditions, sort_keys=True)
        results_json = json.dumps(results)

        # Deterministic exp_id from params
        h = hashlib.md5(params_json.encode()).hexdigest()[:8]
        exp_id = f"{model_id}_{h}"

        # 1) Compute normalized numeric vector
        try:
            vec = self._compute_vector(model_id, params)
            vec_json = json.dumps(vec)
        except Exception as e:
            logger.warning(f"Could not compute numeric vector: {e}")
            vec_json = None

        # 2) Compute joint embedding
        try:
            spec = self.get_model(model_id)["spec"]
            param_ranges = spec.get("param_ranges", {})
            exp_emb = embed_experiment(
                params=params,
                ics=initial_conditions,
                results=results,
                model_id=model_id,
                param_ranges=param_ranges
            )
            emb_json = json.dumps(exp_emb)
        except Exception as e:
            logger.warning(f"Failed to compute experiment embedding: {e}")
            emb_json = None

        # 3) Insert into DB
        cur.execute("""
        INSERT OR REPLACE INTO experiments
          (exp_id, model_id, params, initial_conditions, vector, results, embedding, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            exp_id,
            model_id,
            params_json,
            ics_json,
            vec_json,
            results_json,
            emb_json,
            now
        ))
        self.conn.commit()
        logger.info(f"Stored experiment {exp_id} for model {model_id}")

    def get_experiments(self, exp_id: str):
        cur = self.conn.cursor()
        cur.execute("""
        SELECT id, params, results, created_at FROM experiments
        WHERE model_id = ?
        ORDER BY id
        """, (exp_id,))
        rows = cur.fetchall()
        out = []
        for row in rows:
            out.append({
                "id": row["id"],
                "params": json.loads(row["params"]),
                "results": json.loads(row["results"]),
                "created_at": row["created_at"]
            })
        return out

    def find_experiment(self, model_id: str, params: dict):
        """
        Check if an experiment with exactly these params exists.
        NOTE: JSON text matching; ordering matters. For robust matching, normalize keys sorted.
        """
        # Normalize JSON: sort keys
        params_json = json.dumps(params, sort_keys=True)
        cur = self.conn.cursor()
        # We stored params without sorting; so we should fetch all and compare normalized
        cur.execute("""
        SELECT exp_id, params, results, created_at FROM experiments
        WHERE model_id = ?
        """, (model_id,))
        for row in cur.fetchall():
            stored = json.loads(row["params"])
            if json.dumps(stored, sort_keys=True) == params_json:
                return {
                    "id": row["id"],
                    "params": stored,
                    "results": json.loads(row["results"]),
                    "created_at": row["created_at"]
                }
        return None

    # datastore.py (inside DataStore class)

    def set_approved(self, model_id: str, is_approved: bool):
        cur = self.conn.cursor()
        cur.execute("UPDATE models SET approved = ? WHERE model_id = ?",
                    (1 if is_approved else 0, model_id))
        self.conn.commit()

    def is_approved(self, model_id: str) -> bool:
        cur = self.conn.cursor()
        cur.execute("SELECT approved FROM models WHERE model_id = ?", (model_id,))
        row = cur.fetchone()
        return bool(row["approved"]) if row else False

