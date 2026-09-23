from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


FILE_PATH = Path("data/documents/ai_engineering.txt")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_document(file_path: Path) -> list[Document]:

    text = file_path.read_text(encoding="utf-8")

    document = Document(
        page_content=text,
        metadata={
            "source": str(file_path),
            "file_name": file_path.name,
        },
    )

    return [document]


def chunk_documents(documents: list[Document]) -> list[Document]:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )

    return splitter.split_documents(documents)


if __name__ == "__main__":

    # -------------------------
    # 1. Load
    # -------------------------

    documents = load_document(FILE_PATH)

    # -------------------------
    # 2. Chunk
    # -------------------------

    chunks = chunk_documents(documents)

    print("Number of chunks:", len(chunks))

    # -------------------------
    # 3. Load embedding model
    # -------------------------

    model = SentenceTransformer(MODEL_NAME)

    # -------------------------
    # 4. Extract chunk text
    # -------------------------

    chunk_texts = [
        chunk.page_content
        for chunk in chunks
    ]

    # -------------------------
    # 5. Embed chunks
    # -------------------------

    chunk_embeddings = model.encode(chunk_texts)

    print(
        "Chunk embeddings shape:",
        chunk_embeddings.shape
    )
    query = "What caused INC-552104?"

    query_embedding = model.encode(query)

    print(
    "Query embedding shape:",
    query_embedding.shape
    )

    similarities = model.similarity(
    query_embedding,
    chunk_embeddings
    )

    print("\nSimilarity scores:")
    print(similarities)

    TOP_K=3

    scores=similarities[0]

    ranked_indices=scores.argsort(descending=True)

    top_indices=ranked_indices[:TOP_K]

    print("\nTOP RESULTS")
    print("=" * 70)

    for rank, index in enumerate(top_indices, start=1):

      index = index.item()

      print(f"\nRANK {rank}")
      print(f"Chunk index: {index}")
      print(f"Similarity: {scores[index]:.4f}")

      print("\nContent:")
      print(chunks[index].page_content)

      print("-" * 70)