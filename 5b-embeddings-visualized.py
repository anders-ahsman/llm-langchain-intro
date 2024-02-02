from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from langchain_openai.embeddings import OpenAIEmbeddings
from sklearn.manifold import TSNE

from common import EXAMPLE_TEXTS

EMBEDDINGS_FILENAME = "embeddings.npz"


def main():
    embeddings, texts = get_embeddings_and_texts()

    print(f"Number of embeddings: {len(embeddings)}")

    visualize_embeddings(embeddings, texts)


def get_embeddings_and_texts() -> Tuple[np.ndarray, list[str]]:
    """
    Try to load embeddings and texts from file.
    If file is not found, compute embeddings and save to file.
    """

    embeddings_model = OpenAIEmbeddings()

    try:
        data = np.load(EMBEDDINGS_FILENAME)
        embeddings = data["embeddings"]
        texts = data["texts"].tolist()  # Convert numpy array of objects back to list
        print("Embeddings and texts loaded from file.")
    except FileNotFoundError:
        print("Embeddings file not found, computing embeddings...")

        # Note: This can be run in parallel for large datasets to speed up computation
        embeddings = np.array([embeddings_model.embed_query(text) for text in EXAMPLE_TEXTS])

        # Save embeddings and texts together
        np.savez(EMBEDDINGS_FILENAME, embeddings=embeddings, texts=EXAMPLE_TEXTS)
        print("Embeddings and texts saved to file.")

    return embeddings, texts


def visualize_embeddings(
    embeddings: np.ndarray,
    texts: list[str],
    n_components=2,
    perplexity=30,
    init="pca",
    random_state=42,
):
    """
    Visualize high-dimensional embeddings using t-SNE and annotate with texts.
    """

    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        init=init,
        random_state=random_state,
    )
    embeddings_reduced = tsne.fit_transform(embeddings)

    plt.figure(figsize=(12, 10))
    for i, (x, y) in enumerate(embeddings_reduced):
        plt.scatter(x, y)
        plt.text(x - 0.02, y + 0.05, texts[i], fontsize=9)

    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.title("2D Visualization of Text Embeddings")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
