from langchain_community.document_loaders import PyMuPDFLoader

PDF_PATH="data/documents/project-notes.pdf"

loader=PyMuPDFLoader(PDF_PATH)

documents=loader.load()

print("Number of pages", len(documents))

print("\n First page:")
print(documents[0].page_content)

print("\nMetadata:")
print(documents[0].metadata)