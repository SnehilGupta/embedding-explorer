import os
from pathlib import Path
from typing import List

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

from utils.convert_jsonl import convert_jsonl_to_json

app = FastAPI(title="Word Embeddings API")

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
SCRIPT_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = SCRIPT_DIR / "uploads"
MODEL_DIR = SCRIPT_DIR / "models"
UPLOAD_DIR.mkdir(exist_ok=True, parents=True)
MODEL_DIR.mkdir(exist_ok=True, parents=True)

# Logging
print(f"[STARTUP] Working directory: {os.getcwd()}")
print(f"[STARTUP] Script directory: {SCRIPT_DIR}")
print(f"[STARTUP] Upload directory: {UPLOAD_DIR} (exists: {UPLOAD_DIR.exists()})")
print(f"[STARTUP] Model directory: {MODEL_DIR} (exists: {MODEL_DIR.exists()})")
print(f"[STARTUP] Model directory writable: {os.access(MODEL_DIR, os.W_OK)}")

# Convert dataset at startup
base_dir = Path(__file__).resolve().parent.parent
input_path = base_dir / "data" / "News_Category_Dataset_v3.json"
output_path = base_dir / "data" / "converted_dataset.json"

if input_path.exists():
    convert_jsonl_to_json(str(input_path), str(output_path))
    print(f"[STARTUP] Converted dataset: {output_path} (exists: {output_path.exists()})")
else:
    print(f"[WARNING] Input dataset not found: {input_path}")

model_status = {}

@app.get("/")
def read_root():
    return {"message": "Word Embeddings API"}

@app.post("/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)):
    try:
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        if not file_path.exists():
            raise HTTPException(status_code=500, detail="File upload failed")

        data = load_dataset(str(file_path))
        return JSONResponse(
            content={
                "message": "Dataset uploaded successfully",
                "filename": file.filename,
                "rows": len(data),
                "columns": list(data.columns),
                "file_size": file_path.stat().st_size
            },
            status_code=200
        )
    except Exception as e:
        print(f"[ERROR] Upload failed: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/train-model")
async def train_model(
    background_tasks: BackgroundTasks,
    model_type: str = Form(...),
    dataset_file: str = Form(...),
    text_column: str = Form(...),
    min_count: int = Form(5),
    vector_size: int = Form(100),
    window: int = Form(5),
    sg: int = Form(0),
):
    model_type_normalized = model_type.replace("-", "").lower()
    if model_type_normalized not in ["tfidf", "word2vec"]:
        raise HTTPException(status_code=400, detail=f"Unsupported model type: {model_type}")

    model_id = f"{model_type_normalized}_{int(time.time())}"
    model_status[model_id] = {"status": "queued", "progress": 0}

    print(f"[TRAIN] Model training initiated with ID: {model_id}")
    print(f"[TRAIN] Parameters: type={model_type_normalized}, file={dataset_file}, column={text_column}")

    background_tasks.add_task(
        _train_model_task,
        model_id,
        model_type_normalized,
        dataset_file,
        text_column,
        min_count,
        vector_size,
        window,
        sg
    )

    return {"model_id": model_id, "status": "queued"}

async def _train_model_task(
    model_id: str,
    model_type: str,
    dataset_file: str,
    text_column: str,
    min_count: int,
    vector_size: int,
    window: int,
    sg: int,
):
    try:
        model_type = model_type.replace("-", "").lower()
        print(f"[TRAINING] Starting training for model: {model_id}")

        file_path = UPLOAD_DIR / dataset_file
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        data = load_dataset(str(file_path))
        corpus = data[text_column].apply(preprocess_text).tolist()
        print(f"[TRAINING] Corpus processed: {len(corpus)} documents")

        if model_type == "tfidf":
            embedder = TFIDFEmbedder()
            embedder.fit(corpus)
        elif model_type == "word2vec":
            embedder = Word2VecEmbedder(vector_size=vector_size, window=window, min_count=min_count, sg=sg)
            embedder.fit(corpus)
        else:
            raise ValueError(f"Invalid model type: {model_type}")

        model_path = MODEL_DIR / f"{model_id}.joblib"
        joblib.dump(embedder, model_path)
        file_size = model_path.stat().st_size

        model_status[model_id] = {"status": "completed", "progress": 100, "file_size": file_size}
        print(f"[TRAINING] Model saved: {model_path} ({file_size} bytes)")

    except Exception as e:
        print(f"[ERROR] Model training failed: {e}")
        print(traceback.format_exc())
        model_status[model_id] = {"status": "failed", "error": str(e)}

@app.get("/model-status/{model_id}")
async def get_model_status(model_id: str):
    if model_id not in model_status:
        raise HTTPException(status_code=404, detail="Model not found")
    return model_status[model_id]

@app.get("/models")
async def list_models():
    try:
        model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".joblib")]
        models = []

        for file in model_files:
            model_id = file.replace(".joblib", "")
            model_type = "tfidf" if model_id.startswith("tfidf") else "word2vec"
            model_path = MODEL_DIR / file
            file_size = model_path.stat().st_size
            models.append({
                "id": model_id,
                "type": model_type,
                "status": model_status.get(model_id, {}).get("status", "completed"),
                "file_size": file_size,
                "created": model_path.stat().st_mtime
            })
        return models
    except Exception as e:
        print(f"[ERROR] Failed to list models: {e}")
        print(traceback.format_exc())
        return []

@app.get("/similar-words/{model_id}")
async def get_similar_words(model_id: str, word: str, top_n: int = 10):
    try:
        model_path = MODEL_DIR / f"{model_id}.joblib"
        if not model_path.exists():
            raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
        embedder = joblib.load(model_path)
        return {"word": word, "similar_words": embedder.get_similar_words(word, top_n)}
    except Exception as e:
        print(f"[ERROR] Similar words failed: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/visualize-embeddings/{model_id}")
async def visualize_embeddings(
        model_id: str,
        words: List[str] = Form(default=[]),  # Changed from None to default=[]
        method: str = Form("pca"),
        n_components: int = Form(2),
        random_state: int = Form(42),
        sample_size: int = Form(200)
):
    try:
        model_path = MODEL_DIR / f"{model_id}.joblib"
        if not model_path.exists():
            raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")

        embedder = joblib.load(model_path)

        # Handle words parameter properly
        if words and len(words) > 0:
            word_vectors = {w: embedder.get_vector(w) for w in words if embedder.has_word(w)}
        else:
            vocab = embedder.get_vocabulary()
            sample = np.random.choice(vocab, size=min(sample_size, len(vocab)), replace=False)
            word_vectors = {w: embedder.get_vector(w) for w in sample if embedder.has_word(w)}

        if not word_vectors:
            raise HTTPException(status_code=400, detail="No valid words found")

        vectors = np.array(list(word_vectors.values()))
        labels = list(word_vectors.keys())
        reduced_vectors = reduce_dimensions(vectors, method, n_components, random_state)

        return {
            "words": labels,
            "vectors": reduced_vectors.tolist(),
            "method": method,
            "model_id": model_id,
            "n_words": len(labels)
        }

    except Exception as e:
        print(f"[ERROR] Visualization failed: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_dir_exists": MODEL_DIR.exists(),
        "model_dir_writable": os.access(MODEL_DIR, os.W_OK),
        "upload_dir_exists": UPLOAD_DIR.exists(),
        "upload_dir_writable": os.access(UPLOAD_DIR, os.W_OK),
        "active_models": len(model_status)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import os
import time
import joblib
import shutil
import traceback
from typing import List
from pathlib import Path

from models.tfidf import TFIDFEmbedder
from models.word2vec import Word2VecEmbedder
from utils.utils import load_dataset, preprocess_text, reduce_dimensions
from utils.convert_jsonl import convert_jsonl_to_json

app = FastAPI(title="Word Embeddings API")

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
SCRIPT_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = SCRIPT_DIR / "uploads"
MODEL_DIR = SCRIPT_DIR / "models"
UPLOAD_DIR.mkdir(exist_ok=True, parents=True)
MODEL_DIR.mkdir(exist_ok=True, parents=True)

# Logging
print(f"[STARTUP] Working directory: {os.getcwd()}")
print(f"[STARTUP] Script directory: {SCRIPT_DIR}")
print(f"[STARTUP] Upload directory: {UPLOAD_DIR} (exists: {UPLOAD_DIR.exists()})")
print(f"[STARTUP] Model directory: {MODEL_DIR} (exists: {MODEL_DIR.exists()})")
print(f"[STARTUP] Model directory writable: {os.access(MODEL_DIR, os.W_OK)}")

# Convert dataset at startup
base_dir = Path(__file__).resolve().parent.parent
input_path = base_dir / "data" / "News_Category_Dataset_v3.json"
output_path = base_dir / "data" / "converted_dataset.json"

if input_path.exists():
    convert_jsonl_to_json(str(input_path), str(output_path))
    print(f"[STARTUP] Converted dataset: {output_path} (exists: {output_path.exists()})")
else:
    print(f"[WARNING] Input dataset not found: {input_path}")

model_status = {}

@app.get("/")
def read_root():
    return {"message": "Word Embeddings API"}

@app.post("/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)):
    try:
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        if not file_path.exists():
            raise HTTPException(status_code=500, detail="File upload failed")

        data = load_dataset(str(file_path))
        return JSONResponse(
            content={
                "message": "Dataset uploaded successfully",
                "filename": file.filename,
                "rows": len(data),
                "columns": list(data.columns),
                "file_size": file_path.stat().st_size
            },
            status_code=200
        )
    except Exception as e:
        print(f"[ERROR] Upload failed: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/train-model")
async def train_model(
    background_tasks: BackgroundTasks,
    model_type: str = Form(...),
    dataset_file: str = Form(...),
    text_column: str = Form(...),
    min_count: int = Form(5),
    vector_size: int = Form(100),
    window: int = Form(5),
    sg: int = Form(0),
):
    model_type_normalized = model_type.replace("-", "").lower()
    if model_type_normalized not in ["tfidf", "word2vec"]:
        raise HTTPException(status_code=400, detail=f"Unsupported model type: {model_type}")

    model_id = f"{model_type_normalized}_{int(time.time())}"
    model_status[model_id] = {"status": "queued", "progress": 0}

    print(f"[TRAIN] Model training initiated with ID: {model_id}")
    print(f"[TRAIN] Parameters: type={model_type_normalized}, file={dataset_file}, column={text_column}")

    background_tasks.add_task(
        _train_model_task,
        model_id,
        model_type_normalized,
        dataset_file,
        text_column,
        min_count,
        vector_size,
        window,
        sg
    )

    return {"model_id": model_id, "status": "queued"}

async def _train_model_task(
    model_id: str,
    model_type: str,
    dataset_file: str,
    text_column: str,
    min_count: int,
    vector_size: int,
    window: int,
    sg: int,
):
    try:
        model_type = model_type.replace("-", "").lower()
        print(f"[TRAINING] Starting training for model: {model_id}")

        file_path = UPLOAD_DIR / dataset_file
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        data = load_dataset(str(file_path))
        corpus = data[text_column].apply(preprocess_text).tolist()
        print(f"[TRAINING] Corpus processed: {len(corpus)} documents")

        if model_type == "tfidf":
            embedder = TFIDFEmbedder()
            embedder.fit(corpus)
        elif model_type == "word2vec":
            embedder = Word2VecEmbedder(vector_size=vector_size, window=window, min_count=min_count, sg=sg)
            embedder.fit(corpus)
        else:
            raise ValueError(f"Invalid model type: {model_type}")

        model_path = MODEL_DIR / f"{model_id}.joblib"
        joblib.dump(embedder, model_path)
        file_size = model_path.stat().st_size

        model_status[model_id] = {"status": "completed", "progress": 100, "file_size": file_size}
        print(f"[TRAINING] Model saved: {model_path} ({file_size} bytes)")

    except Exception as e:
        print(f"[ERROR] Model training failed: {e}")
        print(traceback.format_exc())
        model_status[model_id] = {"status": "failed", "error": str(e)}

@app.get("/model-status/{model_id}")
async def get_model_status(model_id: str):
    if model_id not in model_status:
        raise HTTPException(status_code=404, detail="Model not found")
    return model_status[model_id]

@app.get("/models")
async def list_models():
    try:
        model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".joblib")]
        models = []

        for file in model_files:
            model_id = file.replace(".joblib", "")
            model_type = "tfidf" if model_id.startswith("tfidf") else "word2vec"
            model_path = MODEL_DIR / file
            file_size = model_path.stat().st_size
            models.append({
                "id": model_id,
                "type": model_type,
                "status": model_status.get(model_id, {}).get("status", "completed"),
                "file_size": file_size,
                "created": model_path.stat().st_mtime
            })
        return models
    except Exception as e:
        print(f"[ERROR] Failed to list models: {e}")
        print(traceback.format_exc())
        return []

@app.get("/similar-words/{model_id}")
async def get_similar_words(model_id: str, word: str, top_n: int = 10):
    try:
        model_path = MODEL_DIR / f"{model_id}.joblib"
        if not model_path.exists():
            raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
        embedder = joblib.load(model_path)
        return {"word": word, "similar_words": embedder.get_similar_words(word, top_n)}
    except Exception as e:
        print(f"[ERROR] Similar words failed: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/visualize-embeddings/{model_id}")
async def visualize_embeddings(
    model_id: str,
    words: List[str] = Form(None),
    method: str = Form("pca"),
    n_components: int = Form(2),
    random_state: int = Form(42),
    sample_size: int = Form(200)
):
    try:
        model_path = MODEL_DIR / f"{model_id}.joblib"
        if not model_path.exists():
            raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")

        embedder = joblib.load(model_path)
        if words and len(words) > 0:
            word_vectors = {w: embedder.get_vector(w) for w in words if embedder.has_word(w)}
        else:
            vocab = embedder.get_vocabulary()
            sample = np.random.choice(vocab, size=min(sample_size, len(vocab)), replace=False)
            word_vectors = {w: embedder.get_vector(w) for w in sample if embedder.has_word(w)}

        if not word_vectors:
            raise HTTPException(status_code=400, detail="No valid words found")

        vectors = np.array(list(word_vectors.values()))
        labels = list(word_vectors.keys())
        reduced_vectors = reduce_dimensions(vectors, method, n_components, random_state)

        return {
            "words": labels,
            "vectors": reduced_vectors.tolist(),
            "method": method,
            "model_id": model_id,
            "n_words": len(labels)
        }

    except Exception as e:
        print(f"[ERROR] Visualization failed: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_dir_exists": MODEL_DIR.exists(),
        "model_dir_writable": os.access(MODEL_DIR, os.W_OK),
        "upload_dir_exists": UPLOAD_DIR.exists(),
        "upload_dir_writable": os.access(UPLOAD_DIR, os.W_OK),
        "active_models": len(model_status)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
