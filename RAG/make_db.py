import os
import re
import shutil

from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Load .env from current working tree first, then RAG/.env explicitly.
load_dotenv()
load_dotenv(os.path.join(BASE_DIR, ".env"))

DATA_PATH = os.path.join(BASE_DIR, "data")
CHROMA_PATH = os.path.join(BASE_DIR, "chroma")


# -------------------- Entry point --------------------
def main():
    generate_data_store()


def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)


# -------------------- Load --------------------
def load_documents() -> list[Document]:
    """
    Read every .md file in DATA_PATH and return a list of LangChain Documents,
    one Document per file.
    """
    documents = []

    for filename in os.listdir(DATA_PATH):
        if not filename.endswith(".md"):
            continue

        filepath = os.path.join(DATA_PATH, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        documents.append(
            Document(
                page_content=text,
                metadata={"source": filepath},
            )
        )

    print(f"Loaded {len(documents)} document(s).")
    return documents


# -------------------- Split --------------------
def split_text(documents: list[Document]) -> list[Document]:
    """
    Split each document into one chunk per Q&A pair.

    Expected format in the .md file:
        1. Question text
           Answer text (one or more lines, indented or not)

        2. Next question ...

    Each chunk gets metadata: source, question_number, question
    """
    chunks = []

    # Matches a numbered item and captures everything until the next number or EOF
    qa_pattern = re.compile(
        r"^(\d+)\.\s+(.+?)\n(.*?)(?=^\d+\.\s|\Z)",
        re.MULTILINE | re.DOTALL,
    )

    for doc in documents:
        matches = qa_pattern.findall(doc.page_content)

        for number, question, answer in matches:
            question = question.strip()
            answer = answer.strip()

            # Combine into one clean chunk
            page_content = f"Q: {question}\nA: {answer}"

            chunks.append(
                Document(
                    page_content=page_content,
                    metadata={
                        "source": doc.metadata["source"],
                        "question_number": int(number),
                        "question": question,
                    },
                )
            )

    print(f"Split {len(documents)} document(s) into {len(chunks)} chunk(s).")

    # Quick sanity-check printout
    if chunks:
        sample = chunks[0]
        print("\n--- Sample chunk ------------------------------")
        print(sample.page_content)
        print("Metadata:", sample.metadata)
        print("----------------------------------------------\n")

    return chunks


# -------------------- Save to Chroma --------------------
def save_to_chroma(chunks: list[Document]) -> None:
    """
    Persist chunks to a local Chroma vector store.
    Clears any existing DB first so re-runs are always fresh.
    """
    # Wipe existing DB to avoid duplicates on re-runs
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print(f"Cleared existing Chroma DB at '{CHROMA_PATH}'.")

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Add it to RAG/.env or set it in your environment."
        )

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key,
    )

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
    )

    print(f"Saved {len(chunks)} chunk(s) to Chroma at '{CHROMA_PATH}'.")


if __name__ == "__main__":
    main()
