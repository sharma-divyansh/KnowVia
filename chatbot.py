
from pathlib import Path

import fitz
import ollama
from docx import Document
from sentence_transformers import SentenceTransformer, util

# Find the project folders
BASE_DIR = Path(__file__).parent
DOCUMENTS_DIR = BASE_DIR / "documents"

# Store text passages and their source information
passages = []
sources = []


def add_passages(text, filename, location):
    """Split extracted text into searchable paragraphs."""
    paragraphs = text.split("\n\n")

    for paragraph in paragraphs:
        paragraph = " ".join(paragraph.split()).strip()

        if len(paragraph) > 20:
            passages.append(paragraph)
            sources.append({
                "filename": filename,
                "location": location
            })

def read_txt(file_path):
    text = file_path.read_text(encoding="utf-8-sig")
    add_passages(text, file_path.name, "Text file")


def read_pdf(file_path):
    with fitz.open(file_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text()
            add_passages(
                text,
                file_path.name,
                f"Page {page_number}"
            )


def read_docx(file_path):
    document = Document(file_path)

    for paragraph in document.paragraphs:
        text = paragraph.text.strip()

        if len(text) > 20:
            passages.append(text)
            sources.append({
                "filename": file_path.name,
                "location": "Word document"
            })


def load_documents():
    if not DOCUMENTS_DIR.exists():
        DOCUMENTS_DIR.mkdir(parents=True)
        print("Created the documents folder.")

    # Include the original root document as well
    files = []

    original_file = BASE_DIR / "document.txt"
    if original_file.exists():
        files.append(original_file)

    files.extend(
        file_path
        for file_path in DOCUMENTS_DIR.rglob("*")
        if file_path.is_file()
        and file_path.suffix.lower() in {".txt", ".pdf", ".docx"}
    )

    for file_path in files:
        try:
            suffix = file_path.suffix.lower()

            if suffix == ".txt":
                read_txt(file_path)
            elif suffix == ".pdf":
                read_pdf(file_path)
            elif suffix == ".docx":
                read_docx(file_path)

            print(f"Loaded: {file_path.name}")

        except Exception as error:
            print(f"Could not read {file_path.name}: {error}")


# Load the documents
print("Reading documents...")
load_documents()

if not passages:
    print("No readable text found. Add files to documents.")
    raise SystemExit

print(f"\nLoaded {len(passages)} text passages.")

# Create semantic embeddings
print("Loading the embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Creating document embeddings...")
passage_embeddings = model.encode(
    passages,
    convert_to_tensor=True
)

print("\nDocument Q&A chatbot is ready!")
print("Ask a question or type 'exit' to quit.\n")


while True:
    question = input("You: ").strip()

    if question.lower() in {"exit", "quit"}:
        print("Goodbye!")
        break

    if not question:
        continue

    # Convert the question into an embedding
    question_embedding = model.encode(
        question,
        convert_to_tensor=True
    )

    # Retrieve the three most relevant passages
    count = min(3, len(passages))

    results = util.cos_sim(
        question_embedding,
        passage_embeddings
    )[0].topk(count)

    context_parts = []

    for index, score in zip(
        results.indices.tolist(),
        results.values.tolist()
    ):
        source = sources[index]

        context_parts.append(
            f"[Source: {source['filename']}, "
            f"{source['location']}]\n"
            f"{passages[index]}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
Answer the question using only the source passages below.

Rules:
- Do not invent facts or use outside knowledge.
- If the passages do not answer the question, say so.
- Keep the answer clear and concise.
- Cite the filename and page/location supporting your answer.
- Distinguish stated facts from calculations or assumptions.

Source passages:
{context}

Question:
{question}
"""

    try:
        response = ollama.chat(
            model="llama3.2:3b",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        print("\nBot:", response["message"]["content"])
        print()

    except Exception as error:
        print("\nCould not contact Ollama.")
        print("Check that Ollama is running.")
        print("Error:", error, "\n")
