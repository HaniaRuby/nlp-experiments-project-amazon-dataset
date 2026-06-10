# scripts/visualize_embeddings_3d.py
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

print("Generating synthetic 3D dense word embeddings based on conceptual vectors...")

# Define structural semantic clusters to demonstrate vector alignment
words = [
    # Cluster 1: Royalty & Gender (Red)
    "king", "queen", "man", "woman", "prince", "princess",
    # Cluster 2: Beverage/Roast Domain (Blue)
    "coffee", "tea", "espresso", "beverage", "roast", "drink",
    # Cluster 3: Positive Sentiment (Green)
    "amazing", "delicious", "loved", "great", "excellent", "awesome",
    # Cluster 4: Negative Sentiment (Purple)
    "terrible", "stale", "bad", "horrible", "awful", "annoying"
]

# Build deterministic high-dimensional representations (30 dimensions)
np.random.seed(42)
embedding_dim = 30
vectors = np.zeros((len(words), embedding_dim))

# Define clean cluster centers for structured semantic offsets
centers = {
    "royalty": np.array([3.0 if i < 10 else 0.0 for i in range(embedding_dim)]),
    "beverage": np.array([0.0 if i < 10 else (3.0 if i < 20 else 0.0) for i in range(embedding_dim)]),
    "positive": np.array([0.0 if i < 20 else 3.0 for i in range(embedding_dim)]),
    "negative": np.array([0.0 if i < 20 else -3.0 for i in range(embedding_dim)])
}

# Distribute words with minimal Gaussian variation to maintain tight grouping
for idx, word in enumerate(words):
    if idx < 6: base = centers["royalty"]
    elif idx < 12: base = centers["beverage"]
    elif idx < 18: base = centers["positive"]
    else: base = centers["negative"]
    vectors[idx] = base + np.random.normal(0, 0.1, embedding_dim)

print("Compressing high-dimensional space down to 3D using t-SNE...")
# Set n_components=3 for a spatial representation
tsne = TSNE(n_components=3, perplexity=5, random_state=42, n_iter=1000)
vectors_3d = tsne.fit_transform(vectors)

# Create the 3D plot visualization canvas
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# Grouping colors for clarity (matches 2D version)
colors = ['red']*6 + ['blue']*6 + ['green']*6 + ['purple']*6

print("Generating 3D spatial neighborhood map...")
for i, word in enumerate(words):
    # Plot the point in X, Y, Z space
    ax.scatter(vectors_3d[i, 0], vectors_3d[i, 1], vectors_3d[i, 2], color=colors[i], s=150, alpha=0.8)
    # Annotate text with bolding for core concepts
    ax.text(
        vectors_3d[i, 0] + 0.3, 
        vectors_3d[i, 1] + 0.3, 
        vectors_3d[i, 2] + 0.3, 
        word, 
        fontsize=11,
        weight='bold' if word in ["coffee", "king", "queen", "amazing", "terrible"] else 'normal'
    )

ax.set_title("3D t-SNE space of Distributional Word Embeddings", fontsize=14, pad=15, weight='bold')
ax.set_xlabel("Latent Semantic Axis X", fontsize=10)
ax.set_ylabel("Latent Semantic Axis Y", fontsize=10)
ax.set_zlabel("Latent Semantic Axis Z", fontsize=10)

# Optimal viewing angle for cluster separation
ax.view_init(elev=20, azim=135)
plt.grid(True, linestyle=':', alpha=0.5)

# Save the conceptual 3D plot map
os.makedirs("plots", exist_ok=True)
plot_out = "plots/conceptual_words_3d_tsne.png"
plt.savefig(plot_out, dpi=300, bbox_inches='tight')
print(f"Success! Clean 3D visualization plot saved to: {plot_out}")