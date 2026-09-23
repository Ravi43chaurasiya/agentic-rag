from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)


sentences = [
    "I purchased a car yesterday.",
    "I bought an automobile yesterday.",
    "A banana is a yellow fruit.",
    "The weather is sunny today.",
]


embeddings = model.encode(sentences)


print("Number of sentences:", len(sentences))
print("Embedding shape:", embeddings.shape)

print("\nFirst 10 dimensions:")
print(embeddings[0][:10])


similarities = model.similarity(embeddings, embeddings)


print("\nSimilarity matrix:")
print(similarities)