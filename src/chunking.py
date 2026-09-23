from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

FILE_PATH=Path("data/documents/ai_engineering.txt")

def load_text_file(file_path:Path)->list[Document]:
  text=file_path.read_text(encoding="utf-8")

  document=Document(
    page_content=text,
    metadata={
      "source":str(file_path),
      "file_name":file_path.name
    },
  )
  return [document]

def chunk_documents(documents: list[Document])->list[Document]:
  splitter=RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
  )
  chunks=splitter.split_documents(documents)
  return chunks

if __name__ == "__main__":

  documents=load_text_file(FILE_PATH)
  chunks=chunk_documents(documents)

  print("Documents:",len(documents))
  print("chunks:",len(chunks))

  for i, chunk in enumerate(chunks):
    print(f"\n{'=' * 60}")
    print(f"CHUNK {i}")
    print(f"{'=' * 60}")

    print(chunk.page_content)

    print("\nMetadata:")
    print(chunk.metadata)
