Word Embeddings Explorer

📚 Word Embeddings Explorer is a full-stack demo that lets you upload a text dataset, train TF-IDF or Word2Vec models, and explore the resulting embeddings through both a REST API and a Streamlit front-end.
Table of contents

    Overview

    Features

    Project structure

    Quick start

    REST API reference

    Streamlit usage

    Development & testing

    Contributing

    License

1 Overview

The application is split into two loosely coupled layers:
Layer	Stack	Responsibilities
Back-end	FastAPI, Gensim, scikit-learn, NumPy, Pandas	- Accept dataset uploads
- Train Word2Vec or TF-IDF models asynchronously
- Persist trained models (.joblib) and expose inference endpoints
Front-end	Streamlit, Plotly	- Provide an easy UI for uploading data, launching training jobs, querying similar words and visualising the embedding space

Communication between the layers happens over simple JSON endpoints; the front-end can therefore be replaced or extended without touching the training logic.
2 Features

    Dataset upload — supports CSV, JSON, XLSX/XLS.

    Two embedding algorithms

        TF-IDF (sparse, frequency-based)

        Word2Vec (dense, CBOW or Skip-gram).

    Configurable hyper-parameters (vector size, window, min-count, etc.).

    Asynchronous training — runs in a FastAPI background task, keeping the UI responsive.

    Interactive visualisation — PCA or t-SNE scatter-plots with optional word highlighting.

    Similar-word lookup — cosine similarity for either embedding type.

    Health check & model registry — quickly see which jobs are queued, running, or finished.

3 Project structure

text
.
├─ app.py                 # Streamlit front-end
├─ launcher.py            # Dependency check + Streamlit launcher
├─ backend/
│  ├─ main.py             # FastAPI entry-point
│  ├─ uploads/            # Datasets stored here
│  ├─ models/             # Saved .joblib models
│  ├─ models/
│  │  ├─ tfidf.py         # TFIDFEmbedder class
│  │  └─ word2vec.py      # Word2VecEmbedder class
│  └─ utils/
│     ├─ utils.py         # helpers: load_dataset, reduce_dimensions, etc.
│     └─ convert_jsonl.py # converts JSONL → JSON for sample data
├─ data/                  # Sample datasets
├─ requirements.txt
└─ README.md              # ← you are here

4 Quick start
4.1 Prerequisites

    Python 3.9 or newer

    pip (or pipx/pipenv/poetry)

4.2 Installation

bash
# clone the repo (replace with your remote)
git clone <your-repo> word-embeddings-explorer
cd word-embeddings-explorer

# install dependencies
python -m pip install -r requirements.txt

4.3 Run the back-end

bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

4.4 Run the front-end

Open a new terminal:

bash
python launcher.py      # auto-checks deps and runs `streamlit run app.py`

The Streamlit app will open in your browser; use the sidebar to navigate.
5 REST API reference
Method	Path	Description	Body / Query
POST	/upload-dataset	Upload a dataset file	multipart-form field file
POST	/train-model	Start a training job	model_type, dataset_file, text_column, plus hyper-params
GET	/model-status/{model_id}	Poll training progress	–
GET	/models	List all saved models	–
GET	/similar-words/{model_id}	Retrieve top-N neighbours	query word, top_n
POST	/visualize-embeddings/{model_id}	Dimensionality-reduction for plotting	form method, n_components, sample_size, optional words[]
GET	/health	Basic health report	–

Example curl snippet:

bash
curl -F "file=@data/sample.csv" /upload-dataset
curl -X POST -F model_type=tfidf -F dataset_file=sample.csv -F text_column=text /train-model

6 Streamlit usage

    Select Dataset → choose a CSV/JSON/Excel file placed in the data/ folder.

    Train Model → pick TF-IDF or Word2Vec, tweak parameters, click Train.

    Explore Embeddings

        Word Similarity tab: enter a word, get neighbours and a local neighbourhood plot.

        Embedding Visualisation tab: generate a PCA or t-SNE scatter-plot for a random vocabulary sample or your own word list.

7 Development & testing

bash
# lint
flake8 .

# run type checks
mypy .

# unit tests (pytest recommended; tests folder not yet included)
pytest -q

Hot-reloading

    Back-end: uvicorn ... --reload already enabled.

    Front-end: Streamlit auto-reloads on file save.

8 Contributing

    Fork & clone.

    Create a feature branch.

    Commit in logical chunks with clear messages.

    Run pytest and ensure flake8 passes.

    Open a pull request against main.

9 License

This project is released under the MIT License. See the LICENSE file for details.
