import streamlit as st
import numpy as np
import networkx as nx
from tqdm import tqdm
from pyvis.network import Network
import streamlit.components.v1 as components
from semvecpy.vectors.semvec_utils import readfile, get_k_vec_neighbors, pathfinder
from semvecpy.vectors.vector_utils import cosine_similarity
import os
import tempfile

# Helper function to get vector for a term
def get_vector_for_term(vectors, term):
    """Get the vector for a specific term"""
    if term in vectors[0]:
        index = vectors[0].index(term)
        return vectors[1][index]
    return None

# Helper function to compute normalized sum of vectors for multiple terms
def get_vector_sum_for_terms(vectors, terms):
    """Get the normalized sum of vectors for multiple terms"""
    term_vectors = []

    for term in terms:
        term = term.strip()  # Remove whitespace
        vector = get_vector_for_term(vectors, term)
        if vector is not None:
            term_vectors.append(vector)

    if not term_vectors:
        return None

    # Sum all vectors
    sum_vector = np.sum(term_vectors, axis=0)

    # Normalize the vector
    norm = np.linalg.norm(sum_vector)
    if norm > 0:
        normalized_vector = sum_vector / norm
    else:
        normalized_vector = sum_vector

    return normalized_vector

# Set page config
st.set_page_config(
    page_title="Pathfinder Network Visualization",
    page_icon="🌐",
    layout="wide"
)

# Title and description
st.title("📊 Pathfinder Network Visualization")
st.markdown("""
This application visualizes semantic neighborhoods using the Pathfinder algorithm.
Enter a term below to see its semantic neighbors and the pruned network.
""")

# Available embedding files from Zenodo (ends with .bin, not .w2v.bin)
# Only working files are listed - positional/proximity files return 404
ZENODO_FILES = [
    "r2_basic_embeddingvectors.bin",
    "r5_basic_embeddingvectors.bin",
    "r2_fasttext_embeddingvectors.bin",
    "r5_fasttext_embeddingvectors.bin",
]

# Helper function to get file label for display
def get_file_label(filename):
    """Extract a readable label from filename"""
    # Remove .bin extension
    name = filename.replace('.bin', '')
    # Format for display: r5_basic -> "R5 (Window=2) Basic", r2_fasttext -> "R2 (Window=2) FastText"
    parts = name.split('_')
    if len(parts) >= 3:
        prefix = parts[0].upper() if parts[0] else ''
        dim = parts[1].upper() if parts[1] else ''
        kind = parts[2].replace('embeddingvectors', '').title() if len(parts) > 2 else ''
        # Determine window size from prefix (r2=window 2, r5=window 5, sw_r2=window 2, sw_r5=window 5)
        window_size = dim[-1] if dim else ''
        label = f"R{window_size} (Window={window_size}) {kind}" if window_size and kind else name.replace('_', ' ').title()
        return label
    return name.replace('_', ' ').title()


def validate_embedding_file(file_path):
    """
    Validate that a downloaded file is a valid embedding vectors file
    Returns (is_valid, error_message, vector_info)
    """
    if not os.path.exists(file_path):
        return False, "File does not exist", None

    try:
        # Check file size (should be substantial for embedding vectors)
        file_size = os.path.getsize(file_path)
        min_expected_size = 1000  # At least 1KB for a valid embedding file
        if file_size < min_expected_size:
            return False, f"File too small ({file_size} bytes), expected at least {min_expected_size} bytes", None

        # Try to read the file and validate its structure
        from semvecpy.vectors.semvec_utils import readfile

        vectors = readfile(file_path)

        # Validate the structure of returned vectors
        if vectors is None:
            return False, "Failed to read embedding vectors from file", None

        if not isinstance(vectors, (list, tuple)) or len(vectors) < 2:
            return False, "Invalid vector file structure: expected (words, vectors)", None

        words = vectors[0]
        vector_data = vectors[1]

        # Check that we have words
        if not isinstance(words, list) or len(words) == 0:
            return False, "No words found in embedding file", None

        # Check that we have vectors
        if not isinstance(vector_data, (list, tuple)) or len(vector_data) == 0:
            return False, "No vectors found in embedding file", None

        # Check that vectors are numeric arrays
        import numpy as np
        first_vector = vector_data[0]
        if not isinstance(first_vector, (list, tuple, np.ndarray)):
            return False, f"Invalid vector format: expected array, got {type(first_vector)}", None

        # Get dimension from first vector
        if isinstance(first_vector, np.ndarray):
            vector_dim = first_vector.shape[0] if len(first_vector.shape) > 0 else 0
        else:
            vector_dim = len(first_vector)

        if vector_dim == 0:
            return False, "Vectors have zero dimension", None

        info = {
            'num_words': len(words),
            'vector_dim': vector_dim,
            'sample_word': words[0] if words else None
        }

        return True, None, info

    except ImportError as e:
        return False, f"Failed to import required module: {str(e)}", None
    except Exception as e:
        return False, f"Error reading embedding file: {str(e)}", None

# Progress bar class for urllib downloads
class TqdmUpTo(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        return self.update(b * bsize - self.n)

# Load test data
@st.cache_data
def load_data_default():
    """Load the test vector files - default version"""
    import urllib.request
    import tempfile

    temp_file = None
    try:
        # Check if the user's embedding file exists locally
        local_embedding_file = "sw_r5_basic_embeddingvectors.bin"
        if os.path.exists(local_embedding_file):
            # Validate local embedding file
            is_valid, error_msg, info = validate_embedding_file(local_embedding_file)
            if not is_valid:
                st.error(f"Local embedding file validation failed: {error_msg}")
                return None, None, None
            st.success(f"Loaded local embedding: {info['num_words']:,} words, {info['vector_dim']} dimensions")
            # Load the user's embedding vector file
            embedding_vectors = readfile(local_embedding_file)
            return embedding_vectors, None, None
        else:
            # Download the default file from Zenodo with progress
            zenodo_url = "https://zenodo.org/records/1345333/files/r5_basic_embeddingvectors.bin?download=1"
            # Use a temp file for download
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".bin")
            temp_file.close()
            temp_path = temp_file.name

            with TqdmUpTo(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc="Downloading vectors") as pbar:
                urllib.request.urlretrieve(zenodo_url, temp_path, reporthook=pbar.update_to)

            # Validate the downloaded file
            is_valid, error_msg, info = validate_embedding_file(temp_path)
            if not is_valid:
                st.error(f"Downloaded file validation failed: {error_msg}")
                os.remove(temp_path)
                return None, None, None

            # Success - show info
            st.success(f"Downloaded and validated: {info['num_words']:,} words, {info['vector_dim']} dimensions")

            # Load and clean up
            embedding_vectors = readfile(temp_path)
            os.remove(temp_path)
            return embedding_vectors, None, None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        if temp_file and os.path.exists(temp_path):
            os.remove(temp_path)
        return None, None, None

@st.cache_data
def load_data_from_zenodo(selected_file):
    """Load the test vector files from a specific Zenodo file"""
    import urllib.request
    import tempfile

    temp_path = None
    try:
        download_url = f"https://zenodo.org/records/1345333/files/{selected_file}?download=1"
        # Use a temp file for download
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".bin")
        temp_file.close()
        temp_path = temp_file.name

        with TqdmUpTo(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=f"Downloading {selected_file}") as pbar:
            urllib.request.urlretrieve(download_url, temp_path, reporthook=pbar.update_to)

        # Validate the downloaded file
        is_valid, error_msg, info = validate_embedding_file(temp_path)
        if not is_valid:
            st.error(f"Downloaded file validation failed: {error_msg}")
            os.remove(temp_path)
            return None, None, None

        # Success - show info
        st.success(f"Downloaded and validated: {info['num_words']:,} words, {info['vector_dim']} dimensions")

        # Load and clean up
        embedding_vectors = readfile(temp_path)
        os.remove(temp_path)
        return embedding_vectors, None, None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        return None, None, None

# Sidebar inputs
st.sidebar.header("Configuration")

# Search term input
search_term = st.sidebar.text_input("Enter search term(s) - separate multiple terms with commas:", value="semantics")

# Number of neighbors
k_neighbors = st.sidebar.slider("Number of neighbors to consider:", min_value=5, max_value=100, value=20)

# Pathfinder parameters
# Set q to default as neighbors - 1 (max = neighbors - 1)
max_q = k_neighbors - 1 if k_neighbors > 1 else 1
default_q = max_q  # Default to maximum possible q
q = st.sidebar.slider("Pathfinder q parameter (max path length):", min_value=1, max_value=max_q, value=default_q)
# For r parameter, we'll use a special case for infinity
r_value = st.sidebar.selectbox("Pathfinder r parameter (distance metric):",
                              options=[1, 2, 3, 4, 5, "infinity"],
                              index=5,  # Default to infinity
                              format_func=lambda x: "infinity" if x == "infinity" else str(x))
# Convert to float if not infinity
if r_value == "infinity":
    r = float('inf')
else:
    r = float(r_value)

# Embedding source selection (must be before button for values to be available)
st.sidebar.markdown("---")
st.sidebar.subheader("Embedding Source")
embedding_source = st.sidebar.radio(
    "Select embedding source:",
    options=["default", "zenodo"],
    format_func=lambda x: "Default (sw_r5_basic)" if x == "default" else "Zenodo (select file)",
    index=0
)

if embedding_source == "zenodo":
    selected_file = st.sidebar.selectbox(
        "Select Zenodo file:",
        options=ZENODO_FILES,
        format_func=get_file_label,
        index=1  # Default to r5_basic (now at index 1)
    )

# Process button (at top of sidebar)
if st.sidebar.button("Visualize Network"):
    # Determine which data to load
    if embedding_source == "zenodo":
        semantic_vectors = load_data_from_zenodo(selected_file)[0]
    else:
        semantic_vectors = load_data_default()[0]
    if not search_term:
        st.warning("Please enter a search term")
        st.stop()

    # Parse multiple terms (comma-separated)
    terms = [term.strip() for term in search_term.split(',') if term.strip()]

    # Find neighbors for the combined vector
    with st.spinner(f"Finding neighbors for terms: {', '.join(terms)}..."):
        # For multiple terms, we compute the sum of their vectors
        if len(terms) == 1:
            # Single term case - use existing logic
            neighbors = get_k_vec_neighbors(semantic_vectors, terms[0], k_neighbors)
            if neighbors is None:
                st.error(f"Term '{terms[0]}' not found in the semantic vector space")
                st.stop()
            search_vector = None  # Will use the original term
        else:
            # Multiple terms - compute vector sum
            sum_vector = get_vector_sum_for_terms(semantic_vectors, terms)
            if sum_vector is None:
                st.error(f"No valid terms found: {', '.join(terms)}")
                st.stop()

            # Find neighbors for the sum vector
            # We need to get the k nearest neighbors to this sum vector
            # Since we don't have a direct function for this, we'll compute similarities manually
            similarities = []
            for i, term in enumerate(tqdm(semantic_vectors[0], desc="Computing similarities")):
                vector = semantic_vectors[1][i]
                # Compute cosine similarity between sum vector and current vector
                similarity = cosine_similarity(sum_vector, vector)
                similarities.append((similarity, term))

            # Sort by similarity and get top k
            similarities.sort(reverse=True)
            neighbors = similarities[:k_neighbors]
            search_vector = sum_vector  # For display purposes

    # Extract neighbor terms and vectors
    neighbor_terms = [neighbor[1] for neighbor in neighbors]

    # Handle both single and multiple term cases for vectors
    if len(terms) == 1:
        neighbor_vectors = [semantic_vectors[1][semantic_vectors[0].index(term)] for term in neighbor_terms]
    else:
        # For multiple terms, we compute the neighbor vectors from the semantic space
        neighbor_vectors = [semantic_vectors[1][semantic_vectors[0].index(term)] for term in neighbor_terms]

    # Compute pairwise similarities with progress bar
    n = len(neighbor_vectors)
    similarity_matrix = np.zeros((n, n))

    with st.spinner("Computing similarity matrix..."):
        for i in tqdm(range(n), desc="Similarity matrix", leave=False):
            for j in range(n):
                if i == j:
                    similarity_matrix[i][j] = 1.0
                else:
                    similarity_matrix[i][j] = cosine_similarity(neighbor_vectors[i], neighbor_vectors[j])

    # Apply Pathfinder algorithm
    with st.spinner("Applying Pathfinder algorithm..."):
        pruned_matrix = pathfinder(q, r, similarity_matrix, cosines=True)

    # Create networkx graph
    G = nx.Graph()

    # Add nodes
    for i, term in enumerate(neighbor_terms):
        G.add_node(i, label=term)

    # Add edges based on pruned matrix
    for i in range(n):
        for j in range(i+1, n):
            if pruned_matrix[i][j] > 0:  # If there's a connection
                G.add_edge(i, j, weight=pruned_matrix[i][j])

    # Create visualization first (at the top)
    st.subheader(f"Semantic Neighborhood for '{search_term}'")

    # Create the network visualization first
    if len(G.nodes()) > 0:
        # Get positions for nodes using spring layout with wider spread
        pos = nx.spring_layout(G, k=2, iterations=100, seed=42)

        # Dynamic canvas height based on number of nodes
        base_height = 400
        node_count = len(G.nodes())
        dynamic_height = base_height + (node_count - 5) * 30  # Add 30px per extra node
        dynamic_height = max(dynamic_height, 400)  # Minimum 400px
        dynamic_height = min(dynamic_height, 800)  # Maximum 800px

        # Create pyvis network with dynamic height
        net = Network(height=f"{dynamic_height}px", width="100%", bgcolor="white", font_color="black")
        net.set_options("""{
            "nodes": {
                "font": {"size": 14, "color": "black", "face": "Arial"},
                "borderWidth": 2,
                "borderWidthSelected": 3,
                "shadow": {"enabled": true, "color": "rgba(0,0,0,0.2)", "size": 5, "x": 2, "y": 2},
                "physics": false,
                "shapeProperties": {"borderRadius": 8},
                "widthMin": 80,
                "widthMax": 200
            },
            "edges": {
                "width": 2,
                "color": {"color": "gray", "highlight": "black"},
                "smooth": {"type": "continuous"}
            },
            "physics": {
                "stabilization": {"enabled": true, "iterations": 100}
            },
            "interaction": {"dragNodes": true, "zoomView": true, "hover": true}
        }""")

        # Add nodes with shape specified per-node
        for i, term in enumerate(neighbor_terms):
            node_color = 'yellow' if term in terms else '#E6E6FA'
            x, y = pos[i]
            text_width = len(term) * 9
            node_size = max(text_width + 30, 100)

            # Set border to 3px for cue words (search terms), 2px for others
            border_width = 3 if term in terms else 2

            net.add_node(
                i,
                label=term,
                x=(x + 1) * 250,
                y=(y + 1) * 250,
                color={"background": node_color, "border": "black"},
                size=node_size,
                shape="box",
                borderWidth=border_width,
                borderWidthSelected=3
            )

        # Add edges
        for edge in G.edges():
            net.add_edge(edge[0], edge[1], value=G[edge[0]][edge[1]]['weight'])

        # Generate and display
        html_string = net.generate_html()
        components.html(html_string, height=dynamic_height + 50, scrolling=False)

        # Show network statistics
        st.markdown("## Network Statistics")
        st.write(f"Number of nodes: {len(G.nodes())}")
        st.write(f"Number of edges: {len(G.edges())}")

        # Show some details about the pruned network
        if len(G.edges()) > 0:
            edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
            st.write(f"Average edge weight: {np.mean(edge_weights):.3f}")
            st.write(f"Max edge weight: {np.max(edge_weights):.3f}")
            st.write(f"Min edge weight: {np.min(edge_weights):.3f}")
    else:
        st.warning("No connections found in the pruned network")

    # Display neighbor terms after the network
    st.markdown("## Top Neighbors")
    neighbor_df = [{"Term": neighbor[1], "Similarity": neighbor[0]} for neighbor in neighbors]
    st.table(neighbor_df)

st.markdown("---")
st.markdown("## How it works")
st.markdown("""
1. **Vector Search**: Finds the k most similar terms to your search term using vector similarities
2. **Similarity Matrix**: Computes pairwise cosine similarities between all neighbors
3. **Pathfinder Algorithm**: Prunes the network based on the q and r parameters to preserve only the most important connections
4. **Visualization**: Displays the pruned network with nodes representing terms and edges representing connections
""")