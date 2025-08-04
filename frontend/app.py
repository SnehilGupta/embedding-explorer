import os
import shutil
from typing import Dict, Any

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

# Set up constants
API_URL = "http://localhost:8000"

# Set page configuration
st.set_page_config(
    page_title="Word Embeddings Explorer",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #333;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .stProgress > div > div > div > div {
        background-color: #1E88E5;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_id' not in st.session_state:
    st.session_state.model_id = None
if 'training_status' not in st.session_state:
    st.session_state.training_status = None
if 'visualization_data' not in st.session_state:
    st.session_state.visualization_data = None

# Helper functions
def select_dataset():
    """Select a dataset from the data directory."""
    # Reset session state variables
    st.session_state.dataset_file = None
    st.session_state.dataset = None
    st.session_state.columns = None

    # Define data directory path
    data_dir = "../data"

    # Get list of available datasets
    try:
        datasets = [f for f in os.listdir(data_dir) if f.endswith(('.csv', '.json', '.xlsx', '.xls'))]
    except FileNotFoundError:
        st.error(f"Data directory not found: {data_dir}")
        datasets = []

    if not datasets:
        st.error("No datasets found in the data directory. Please add datasets to the 'data' folder.")
        return False

    # Dataset selection widget
    selected_dataset = st.selectbox(
        "Select a dataset from the data folder",
        options=datasets,
        format_func=lambda x: x.split('/')[-1] if '/' in x else x
    )

    if not selected_dataset:
        return False

    try:
        with st.spinner("Loading dataset..."):
            # Copy the file to the uploads directory for the API to use
            source_path = os.path.join(data_dir, selected_dataset)
            dest_path = f"uploads/{selected_dataset}"

            # Ensure uploads directory exists
            os.makedirs("uploads", exist_ok=True)

            # Copy the file
            shutil.copy2(source_path, dest_path)

            # Get dataset info from the API
            with open(dest_path, "rb") as file_data:
                # Create the files dictionary with tuple {file_name and byte content} for upload
                files = {"file": (selected_dataset, file_data)}
                # Send POST request with the file
                response = requests.post(f"{API_URL}/upload-dataset", files=files)

            #Stores dataset information in session state
            if response.status_code == 200:
                data = response.json()
                st.session_state.dataset_file = selected_dataset
                st.session_state.columns = data["columns"]
                st.success(f"Dataset loaded successfully: {data['rows']} rows")

                # Display dataset preview - first 5 rows
                if selected_dataset.endswith('.csv'):
                    df = pd.read_csv(source_path)
                    st.dataframe(df.head(5))
                elif selected_dataset.endswith('.json'):
                    df = pd.read_json(source_path)
                    st.dataframe(df.head(5))
                elif selected_dataset.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(source_path)
                    st.dataframe(df.head(5))
                return True
            else:
                st.error(f"Error loading dataset: {response.text}")
                return False

    except Exception as e:
        st.error(f"Error processing dataset: {str(e)}")
        return False

def train_model(model_type: str, text_column: str, params: Dict[str, Any]):
    #Validates prerequisites (dataset must be selected)
    if not st.session_state.dataset_file:
        st.error("Please select a dataset first")
        return

    # Prepare form data
    form_data = {
        "model_type": model_type,
        "dataset_file": st.session_state.dataset_file,
        "text_column": text_column,
        **params
    }

    # Train the model
    response = requests.post(f"{API_URL}/train-model", data=form_data)

    if response.status_code == 200:
        data = response.json()
        st.session_state.model_id = data["model_id"]
        st.session_state.training_status = data["status"]
        return True
    else:
        st.error(f"Error training model: {response.text}")
        return False

def check_model_status():
    if not st.session_state.model_id:
        return None

    # Monitors training progress by polling the API
    response = requests.get(f"{API_URL}/model-status/{st.session_state.model_id}")

    # Updates session state with current training status
    if response.status_code == 200:
        status_data = response.json()
        st.session_state.training_status = status_data["status"]
        return status_data
    else:
        st.error(f"Error checking model status: {response.text}")
        return None

# Finds semantically similar words using trained embeddings
def get_similar_words(word: str, top_n: int = 10):
    if not st.session_state.model_id:
        st.error("Please train a model first")
        return None

    response = requests.get(
        f"{API_URL}/similar-words/{st.session_state.model_id}",
        params={"word": word, "top_n": top_n}
    )

    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error finding similar words: {response.text}")
        return None


def visualize_embeddings(model_id, words=None, method="pca", n_components=2, sample_size=200):
    """Get embedding visualization data"""
    try:
        # Prepare form data
        params = {
            'method': method,
            'n_components': n_components,
            'sample_size': sample_size
        }

        # Add words as query parameters or form data
        if words and len(words) > 0:
            # Send as JSON instead of form data
            payload = {
                **params,
                'words': words  # Send as a list
            }

            response = requests.post(
                f"{API_URL}/visualize-embeddings/{model_id}",
                json=payload  # Send as JSON
            )
        else:
            response = requests.post(
                f"{API_URL}/visualize-embeddings/{model_id}",
                json=params
            )

        response.raise_for_status()
        return response.json()

    except requests.exceptions.HTTPError as e:
        st.error(f"HTTP Error: {e.response.status_code} - {e.response.text}")
        return None
    except Exception as e:
        st.error(f"Visualization failed: {str(e)}")
        return None


def get_models():
    response = requests.get(f"{API_URL}/models")

    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error getting models: {response.text}")
        return []

def plot_embeddings(viz_data: Dict[str, Any], highlight_word: str = None):
    words = viz_data["words"]
    vectors = viz_data["vectors"]

    # Convert to DataFrame for Plotly
    df = pd.DataFrame({
        "word": words,
        "x": [v[0] for v in vectors],
        "y": [v[1] for v in vectors],
        "highlight": [word == highlight_word for word in words] if highlight_word else [False] * len(words)
    })

    # Create scatter plot
    fig = px.scatter(
        df,
        x="x",
        y="y",
        hover_name="word",
        color="highlight",
        color_discrete_map={True: "#FF4B4B", False: "#1E88E5"},
        size=[10 if h else 7 for h in df["highlight"]],
        opacity=0.8,
        title=f"Word Embeddings Visualization ({viz_data['method'].upper()})"
    )

    # Add labels for highlighted word and neighbors
    if highlight_word:
        # Add annotations for highlighted word and similar words
        similar_words_data = get_similar_words(highlight_word, top_n=10)
        if similar_words_data:
            similar_words = [w[0] for w in similar_words_data["similar_words"]]
            for i, word in enumerate(df["word"]):
                if word == highlight_word or word in similar_words[:5]:  # Only show top 5 similar words
                    fig.add_annotation(
                        x=df["x"][i],
                        y=df["y"][i],
                        text=word,
                        showarrow=True,
                        arrowhead=1,
                        ax=0,
                        ay=-30
                    )

    # Update layout
    fig.update_layout(
        autosize=True,
        height=600,
        showlegend=False,
        hovermode="closest",
        xaxis=dict(title=""),
        yaxis=dict(title="")
    )

    return fig

# Main application
def main():
    # Header
    st.markdown("<h1 class='main-header'>Word Embeddings Explorer</h1>", unsafe_allow_html=True)
    st.markdown(
        "An interactive tool to explore and compare TF-IDF and Word2Vec embeddings"
    )

    # Sidebar
    st.sidebar.markdown("## Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Select Dataset", "Train Model", "Explore Embeddings", "About"]
    )

    # Page content
    if page == "Select Dataset":
        st.markdown("<h2 class='sub-header'>Select Dataset</h2>", unsafe_allow_html=True)
        st.write("Select a text dataset from the data folder to train word embedding models.")

        dataset_selected = select_dataset()

        if dataset_selected and st.session_state.columns:
            st.markdown("<div class='info-box'>", unsafe_allow_html=True)
            st.write("### Dataset Information")
            st.write(f"Available columns: {', '.join(st.session_state.columns)}")
            st.markdown("</div>", unsafe_allow_html=True)

    elif page == "Train Model":
        st.markdown("<h2 class='sub-header'>Train Embedding Model</h2>", unsafe_allow_html=True)
        st.write("Select a model type and configure parameters to train an embedding model.")

        if not st.session_state.get("dataset_file"):
            st.warning("Please select a dataset first.")
            st.sidebar.success("Go to 'Select Dataset' to choose your data.")
        else:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("<div class='info-box'>", unsafe_allow_html=True)
                st.write("### Model Configuration")

                model_type = st.selectbox(
                    "Select Model Type",
                    ["TF-IDF", "Word2Vec"],
                    format_func=lambda x: f"{x} (Sparse representation)" if x == "TF-IDF" else f"{x} (Dense representation)"
                )

                if st.session_state.columns:
                    text_column = st.selectbox("Select Text Column", st.session_state.columns)
                else:
                    text_column = st.text_input("Enter Text Column Name")

                # Model-specific parameters
                params = {}

                if model_type == "Word2Vec":
                    st.write("#### Word2Vec Parameters")

                    col_a, col_b = st.columns(2)
                    with col_a:
                        params["vector_size"] = st.slider("Vector Size", min_value=50, max_value=300, value=100, step=10)
                        params["window"] = st.slider("Window Size", min_value=2, max_value=10, value=5, step=1)

                    with col_b:
                        params["min_count"] = st.slider("Minimum Word Count", min_value=1, max_value=10, value=5, step=1)
                        params["sg"] = st.radio(
                            "Algorithm",
                            [0, 1],
                            format_func=lambda x: "CBOW" if x == 0 else "Skip-gram"
                        )

                train_button = st.button("Train Model")
                st.markdown("</div>", unsafe_allow_html=True)

            with col2:
                st.markdown("<div class='info-box'>", unsafe_allow_html=True)
                st.write("### Training Status")

                # Check the current model status
                status_data = check_model_status()

                if train_button:
                    if train_model(model_type.lower(), text_column, params):
                        st.success(f"Model training initiated with ID: {st.session_state.model_id}")
                        status_data = {"status": "queued", "progress": 0}

                if status_data:
                    status = status_data["status"]
                    if status == "completed":
                        st.success("Model training completed successfully!")
                        st.balloons()
                    elif status == "processing":
                        progress = status_data.get("progress", 0)
                        st.warning("Model training in progress...")
                        st.progress(progress / 100)
                    elif status == "queued":
                        st.info("Model training queued...")
                        st.progress(0)
                    elif status == "failed":
                        st.error(f"Model training failed: {status_data.get('error', 'Unknown error')}")

                # List of trained models
                st.write("### Available Models")
                models = get_models()

                if models:
                    for model in models:
                        model_name = model["id"]
                        model_type = "TF-IDF" if "tfidf" in model_name.lower() else "Word2Vec"
                        status = model["status"]

                        col_a, col_b = st.columns([3, 1])
                        with col_a:
                            st.write(f"**{model_name}** - Type: {model_type}, Status: {status}")

                        with col_b:
                            if st.button("Select", key=f"select_{model_name}"):
                                st.session_state.model_id = model_name
                                st.success(f"Selected model: {model_name}")
                else:
                    st.info("No trained models available.")

                st.markdown("</div>", unsafe_allow_html=True)

    elif page == "Explore Embeddings":
        st.markdown("<h2 class='sub-header'>Explore Word Embeddings</h2>", unsafe_allow_html=True)

        if not st.session_state.model_id:
            st.warning("Please train or select a model first.")
            st.sidebar.success("Go to 'Train Model' to create an embedding model.")
        else:
            st.write(f"Currently using model: **{st.session_state.model_id}**")

            tab1, tab2 = st.tabs(["Word Similarity", "Embedding Visualization"])

            with tab1:
                st.markdown("<div class='info-box'>", unsafe_allow_html=True)
                st.write("### Find Similar Words")
                st.write("Enter a word to find semantically similar words based on the embeddings.")

                col_a, col_b = st.columns([3, 1])

                with col_a:
                    query_word = st.text_input("Enter a word", key="similarity_word")

                with col_b:
                    top_n = st.number_input("Number of results", min_value=5, max_value=50, value=10, step=5)

                if st.button("Find Similar Words") and query_word:
                    with st.spinner(f"Finding words similar to '{query_word}'..."):
                        similar_words_data = get_similar_words(query_word, top_n)

                        if similar_words_data:
                            similar_words = similar_words_data["similar_words"]

                            if similar_words and similar_words[0][1] > 0:  # Check if we got valid results
                                st.write(f"### Words similar to '{query_word}'")

                                # Create a DataFrame for the results
                                df = pd.DataFrame(
                                    similar_words,
                                    columns=["Word", "Similarity"]
                                )

                                # Format similarity scores
                                df["Similarity"] = df["Similarity"].apply(lambda x: f"{x:.4f}")

                                # Display as a table
                                st.table(df)

                                # Visualize the word and its neighbors
                                st.write("### Visualization of Similar Words")
                                words_to_visualize = [query_word] + [w[0] for w in similar_words]
                                viz_data = visualize_embeddings(
                                    model_id=st.session_state.model_id,
                                    words=words_to_visualize,
                                    method="pca",
                                    n_components=2
                                )

                                if viz_data:
                                    fig = plot_embeddings(viz_data, highlight_word=query_word)
                                    st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning(f"No similar words found for '{query_word}'. The word might not be in the vocabulary.")
                st.markdown("</div>", unsafe_allow_html=True)

            with tab2:
                st.markdown("<div class='info-box'>", unsafe_allow_html=True)
                st.write("### Embedding Space Visualization")
                st.write("Visualize the embedding space and explore word relationships.")

                col_a, col_b, col_c = st.columns(3)

                with col_a:
                    method = st.selectbox(
                        "Dimensionality Reduction Method",
                        ["PCA", "t-SNE"],
                        format_func=lambda x: f"{x}"
                    )

                with col_b:
                    sample_size = st.slider("Sample Size", min_value=50, max_value=500, value=200, step=50)

                with col_c:
                    highlight_word = st.text_input("Highlight Word (optional)", key="viz_highlight_word")

                if st.button("Visualize Embeddings"):
                    if not st.session_state.model_id:
                        st.error("Please select a model first!")
                    else:
                        with st.spinner("Generating visualization..."):
                            viz_data = visualize_embeddings(
                                model_id=st.session_state.model_id,
                                method=method.lower(),
                                n_components=2,
                                sample_size=sample_size
                            )

                            if viz_data:
                                fig = plot_embeddings(viz_data,
                                                      highlight_word=highlight_word if highlight_word else None)
                                st.plotly_chart(fig, use_container_width=True)

                                st.info(f"Visualizing {viz_data['n_words']} words using {viz_data['method'].upper()}")
                            else:
                                st.error("Failed to generate visualization. Please check your model and try again.")
                st.markdown("</div>", unsafe_allow_html=True)

    elif page == "About":
        st.markdown("<h2 class='sub-header'>About</h2>", unsafe_allow_html=True)

        st.markdown("<div class='info-box'>", unsafe_allow_html=True)
        st.write("""
        ### Word Embeddings Explorer

        This application allows you to explore and compare different word embedding techniques:

        1. **TF-IDF (Term Frequency-Inverse Document Frequency)**
           - Sparse representation
           - Based on word frequency in documents
           - Good for document similarity and search

        2. **Word2Vec**
           - Dense representation
           - Captures semantic relationships between words
           - Two training algorithms:
             - **CBOW (Continuous Bag of Words)**: Predicts a word based on its context
             - **Skip-gram**: Predicts context words based on the current word

        ### How to Use

        1. Upload a text dataset (News Category Dataset recommended)
        2. Train embedding models with your preferred parameters
        3. Explore semantic relationships and visualize the embedding space

        ### Technical Details

        - Frontend: Streamlit
        - Backend: FastAPI
        - Embedding techniques: TF-IDF (scikit-learn) and Word2Vec (gensim)
        - Dimensionality reduction: PCA and t-SNE
        """)
        st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
