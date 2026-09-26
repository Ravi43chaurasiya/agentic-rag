from sentence_transformers import CrossEncoder


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"

reranker = CrossEncoder(MODEL_NAME)


query = "What caused INC-938271?"


documents = [
    """
    Incident INC-552104 occurred in the authentication service.
    The root cause was an expired service credential used by
    the authentication gateway.

    Incident INC-771845 affected the vector database.
    The issue was caused by insufficient memory being allocated
    to the indexing service.
    """,

    """
    Incident INC-938271 occurred in the document processing service.
    The incident was caused by an incorrect parser configuration
    that prevented PDF documents from being processed correctly.
    """,
]


pairs = [
    (query, document)
    for document in documents
]


scores = reranker.predict(pairs)


for index, score in enumerate(scores):

    print(f"\nDocument {index}")
    print(f"Reranker score: {score:.4f}")
    print(documents[index])