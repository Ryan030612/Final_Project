# Final_Project
UMAP-MDPC+
This project implements a combined approach of UMAP for dimensionality reduction and an improved version of MDPC+ (Modified Density Peak Clustering) for unsupervised clustering. It is designed for effective analysis and visualization of high-dimensional data.

# 📁 Project Structure

UMAP-MDPC+/
│
├── main.py                # Main script to run UMAP + MDPC+
├── main1.py               # Alternate entry script or experiments
├── mdpc_umap.py           # Core logic for UMAP + MDPC clustering
├── dpc.py                 # Original DPC (Density Peak Clustering) implementation
├── mdpc_original.py       # Original MDPC implementation
├── evaluation.py          # Clustering evaluation metrics (e.g., ARI, NMI)
├── visual.py              # Visualization functions
├── requirements.txt       # Python dependencies

# 🚀 Quick Start
1. Install Dependencies
We recommend using a virtual environment:

pip install -r requirements.txt

2. Run the Main Script

python main.py

You may also run main1.py to explore alternative configurations or experimental settings.

# 🧠 Algorithm Overview
UMAP (Uniform Manifold Approximation and Projection): A powerful nonlinear dimensionality reduction technique that preserves local topological structure.

MDPC+ (Modified Density Peak Clustering): An improved clustering algorithm based on local density and distance measures, suitable for discovering complex data structures.

# 📊 Output
The program outputs:

Cluster labels

Evaluation metrics such as NMI (Normalized Mutual Information), ARI (Adjusted Rand Index)

Visualizations of clustering results in the reduced 2D space

# ⚙️ Customization
You can modify parameters such as the number of clusters or UMAP configuration directly in main.py.

# 📄 License
This project is intended for academic and research purposes only. Please contact the author for commercial use.
