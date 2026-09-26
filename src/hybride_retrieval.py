from pathlib import Path
import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from sentence_transformers import (
    SentenceTransformer,
    CrossEncoder,
)

FILE_PATH=Path("data/documents/ai_engineering.txt")

MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"

RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"

DENSE_TOP_K=5
BM25_TOP_K=5
FINAL_TOP_K=3

RERANK_TOP_K = 3


RRF_K=60


def load_document(file_path:Path)->list[Document]:

  text=file_path.read_text(encoding="utf-8")

  return [
    Document(
      page_content=text,
      metadata={
        'source':str(file_path),
        'file_name':file_path.name,
      },
    )
  ]

def chunk_documents(documents:list[Document])->list[Document]:

  splitter=RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
  )

  return splitter.split_documents(documents)

def tokenize(text: str) -> list[str]:

    return re.findall(
        r"\b[\w-]+\b",
        text.lower(),
    )

def dense_retrieve(
    query,
    chunks,
    model,
    chunk_embeddings,
    top_k,
):

    query_embedding = model.encode(query)

    similarities = model.similarity(
        query_embedding,
        chunk_embeddings,
    )[0]

    ranked_indices = similarities.argsort(
        descending=True
    )

    results = []

    for index in ranked_indices[:top_k]:

        index = index.item()

        results.append(
            {
                "index": index,
                "score": similarities[index].item(),
            }
        )

    return results

def bm25_retrieve(
    query,
    bm25,
    top_k,
):

    query_tokens = tokenize(query)

    scores = bm25.get_scores(query_tokens)

    ranked_indices = scores.argsort()[::-1]

    results = []

    for index in ranked_indices[:top_k]:

        results.append(
            {
                "index": int(index),
                "score": float(scores[index]),
            }
        )

    return results

def reciprocal_rank_fusion(
    dense_results,
    bm25_results,
    rrf_k=60,
):

    fused_scores = {}

    for results in [dense_results, bm25_results]:

        for rank, result in enumerate(
            results,
            start=1,
        ):

            index = result["index"]

            if index not in fused_scores:
                fused_scores[index] = 0.0

            fused_scores[index] += (
                1 / (rrf_k + rank)
            )

    ranked_results = sorted(
        fused_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    return ranked_results

def rerank(
    query,
    fused_results,
    chunks,
    reranker,
    top_k,
):

    candidate_indices = [
        index
        for index, _ in fused_results
    ]

    pairs = [
        (
            query,
            chunks[index].page_content,
        )
        for index in candidate_indices
    ]

    scores = reranker.predict(pairs)

    reranked_results = []

    for index, score in zip(
        candidate_indices,
        scores,
    ):
        reranked_results.append(
            {
                "index": index,
                "score": float(score),
            }
        )

    reranked_results.sort(
        key=lambda result: result["score"],
        reverse=True,
    )

    return reranked_results[:top_k]


if __name__ == "__main__":

    documents = load_document(FILE_PATH)

    chunks = chunk_documents(documents)

    print("Chunks:", len(chunks))

    # -------------------------
    # Dense index
    # -------------------------

    model = SentenceTransformer(MODEL_NAME)

    reranker = CrossEncoder(
    RERANKER_MODEL_NAME
)

    chunk_texts = [
        chunk.page_content
        for chunk in chunks
    ]

    chunk_embeddings = model.encode(
        chunk_texts
    )

    # -------------------------
    # BM25 index
    # -------------------------

    tokenized_chunks = [
        tokenize(chunk.page_content)
        for chunk in chunks
    ]

    bm25 = BM25Okapi(
        tokenized_chunks
    )

    # -------------------------
    # Query
    # -------------------------

    query = "What caused INC-938271?"

    # -------------------------
    # Retrieve
    # -------------------------

    dense_results = dense_retrieve(
        query,
        chunks,
        model,
        chunk_embeddings,
        DENSE_TOP_K,
    )

    bm25_results = bm25_retrieve(
        query,
        bm25,
        BM25_TOP_K,
    )

    # -------------------------
    # Fuse
    # -------------------------

    fused_results = reciprocal_rank_fusion(
        dense_results,
        bm25_results,
        RRF_K,
    )

    reranked_results = rerank(
    query,
    fused_results,
    chunks,
    reranker,
    RERANK_TOP_K,
)

    # -------------------------
    # Display
    # -------------------------

    print("\nRERANKED RESULTS")
print("=" * 70)

for rank, result in enumerate(
    reranked_results,
    start=1,
):

    index = result["index"]
    score = result["score"]

    print(f"\nRANK {rank}")
    print(f"Chunk: {index}")
    print(f"Reranker score: {score:.4f}")

    print("\nContent:")
    print(chunks[index].page_content)

    print("-" * 70)