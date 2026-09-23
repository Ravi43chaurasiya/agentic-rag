from pathlib import Path
import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi


FILE_PATH = Path("data/documents/ai_engineering.txt")
TOP_K = 3


def load_document(file_path: Path) -> list[Document]:
    text = file_path.read_text(encoding="utf-8")

    return [
        Document(
            page_content=text,
            metadata={
                "source": str(file_path),
                "file_name": file_path.name,
            },
        )
    ]


def chunk_documents(
    documents: list[Document],
) -> list[Document]:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )

    return splitter.split_documents(documents)


def tokenize(text: str) -> list[str]:
    return re.findall(
        r"\b[\w-]+\b",
        text.lower(),
    )


if __name__ == "__main__":

    documents = load_document(FILE_PATH)
    chunks = chunk_documents(documents)

    print("Number of chunks:", len(chunks))

    # Convert every chunk into tokens.
    tokenized_chunks = [
        tokenize(chunk.page_content)
        for chunk in chunks
    ]

    # Build BM25 index.
    bm25 = BM25Okapi(tokenized_chunks)

    query = "How can an agent modify a bad or unclear query?"

    tokenized_query = tokenize(query)

    scores = bm25.get_scores(tokenized_query)

    ranked_indices = scores.argsort()[::-1]

    top_indices = ranked_indices[:TOP_K]

    print("\nQuery:", query)

    print("\nTOP BM25 RESULTS")
    print("=" * 70)

    for rank, index in enumerate(top_indices, start=1):

        print(f"\nRANK {rank}")
        print(f"Chunk index: {index}")
        print(f"BM25 score: {scores[index]:.4f}")

        print("\nContent:")
        print(chunks[index].page_content)

        print("-" * 70)