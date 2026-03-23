import os
import random
import re
import warnings

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

    warnings.filterwarnings(
        "ignore",
        message=r"The class `Chroma` was deprecated in LangChain 0\.2\.9.*",
    )

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data")
CHROMA_PATH = os.path.join(BASE_DIR, "chroma")

load_dotenv()
load_dotenv(os.path.join(BASE_DIR, ".env"))

EVAL_TEMPLATE = """
You are evaluating a student's answer to a technical interview question in {subject}.

Question: {question}
Correct Answer: {correct_answer}
Student's Answer: {student_answer}

Give a short evaluation (2-3 sentences): was the student correct, what did they miss or get right?
Then on a new line write exactly: Score: X/10
where X is a number from 0 to 10.
"""

QA_PATTERN = re.compile(
    r"^(\d+)\.\s+(.+?)\n(.*?)(?=^\d+\.\s|\Z)",
    re.MULTILINE | re.DOTALL,
)

DEFAULT_SUBJECTS = {
    "machine learning": "machine_learning.md",
    "computer networks": "computer_networks.md",
    "data structures and algorithms": "data_structures_and_algorithms.md",
    "object oriented programming basics": "object_oriented_programming_basics.md",
    "artificial intelligence": "artificial_intelligence.md",
}

SUBJECT_ALIASES = {
    "ml": "machine learning",
    "machine learning": "machine learning",
    "machine_learning": "machine learning",
    "cn": "computer networks",
    "computer networks": "computer networks",
    "computer_networks": "computer networks",
    "dsa": "data structures and algorithms",
    "data structures and algorithms": "data structures and algorithms",
    "data_structures_and_algorithms": "data structures and algorithms",
    "oops": "object oriented programming basics",
    "oop": "object oriented programming basics",
    "oops basics": "object oriented programming basics",
    "object oriented programming basics": "object oriented programming basics",
    "object_oriented_programming_basics": "object oriented programming basics",
    "ai": "artificial intelligence",
    "artificial intelligence": "artificial intelligence",
    "artificial_intelligence": "artificial intelligence",
}


def parse_score(evaluation: str) -> int:
    match = re.search(r"Score:\s*(\d+)/10", evaluation)
    return int(match.group(1)) if match else 0


def normalize_subject(subject: str) -> str:
    cleaned = re.sub(r"[^a-z0-9\s]+", " ", subject.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def load_subject_chunks_from_markdown(subject_path: str) -> list[str]:
    with open(subject_path, "r", encoding="utf-8") as file:
        content = file.read()

    chunks: list[str] = []
    for _, question, answer in QA_PATTERN.findall(content):
        page_content = f"Q: {question.strip()}\nA: {answer.strip()}"
        chunks.append(page_content)

    return chunks


def get_subject_path(subject_input: str) -> tuple[str, str]:
    normalized = normalize_subject(subject_input)
    subject_key = SUBJECT_ALIASES.get(normalized, normalized)
    filename = DEFAULT_SUBJECTS.get(subject_key)

    if not filename:
        raise ValueError("Unknown subject")

    subject_path = os.path.join(DATA_PATH, filename)
    if not os.path.exists(subject_path):
        raise FileNotFoundError(f"Missing subject file: {filename}")

    return subject_key, subject_path


def get_random_chunk(chunks: list[str], used_indices: set[int]) -> tuple[int | None, str | None]:
    available = [i for i in range(len(chunks)) if i not in used_indices]
    if not available:
        return None, None
    idx = random.choice(available)
    return idx, chunks[idx]


def parse_chunk(chunk_text: str) -> tuple[str, str]:
    lines = chunk_text.split("\n", 1)
    question = lines[0].replace("Q: ", "").strip()
    correct_ans = lines[1].replace("A: ", "").strip() if len(lines) > 1 else ""
    return question, correct_ans


def try_load_subject_chunks_from_db(db: Chroma | None, subject_path: str) -> list[str]:
    if db is None:
        return []

    try:
        results = db.get(where={"source": subject_path})
    except Exception:
        return []

    return results.get("documents", []) if results else []


def get_followup_chunk(
    db: Chroma | None,
    subject_path: str,
    current_chunk_text: str,
    current_question: str,
    local_chunks: list[str],
    used_indices: set[int],
) -> str | None:
    if db is not None:
        try:
            followup_results = db.similarity_search_with_relevance_scores(current_question, k=8)
        except Exception:
            followup_results = []

        for doc, _ in followup_results:
            if doc.metadata.get("source") != subject_path:
                continue
            if doc.page_content.strip() == current_chunk_text.strip():
                continue
            if doc.page_content in local_chunks:
                idx = local_chunks.index(doc.page_content)
                if idx not in used_indices:
                    used_indices.add(idx)
                    return doc.page_content

    idx, followup_chunk = get_random_chunk(local_chunks, used_indices)
    if idx is None:
        return None
    used_indices.add(idx)
    return followup_chunk


def main() -> None:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Add it to RAG/.env or set it in your environment."
        )

    print("\n----- Subject Quiz -----")
    print("Available subjects:")
    for subject in DEFAULT_SUBJECTS:
        print(f"- {subject.title()}")

    subject_input = input("\nChoose a subject: ").strip()
    try:
        subject_key, subject_path = get_subject_path(subject_input)
    except (ValueError, FileNotFoundError):
        print("\nInvalid subject. Please choose one of the listed subjects.")
        return

    embedding_function = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key,
    )

    db = None
    if os.path.exists(CHROMA_PATH):
        try:
            db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
        except Exception:
            db = None

    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
    )
    eval_prompt = ChatPromptTemplate.from_template(EVAL_TEMPLATE)

    local_chunks = load_subject_chunks_from_markdown(subject_path)
    db_chunks = try_load_subject_chunks_from_db(db, subject_path)
    all_chunks = db_chunks or local_chunks

    if not all_chunks:
        print("No questions found for that subject.")
        return

    if db is None or not db_chunks:
        print("\nUsing the markdown question bank directly for this subject.")
        print("Run make_db.py later if you want Chroma-backed follow-up matching for all subjects.\n")

    print(f"\n----- {subject_key.title()} Quiz -----")
    print("Type 'exit' at any time to stop and see your score.\n")

    used_indices: set[int] = set()
    scores: list[int] = []
    question_num = 0

    while True:
        idx, chunk_text = get_random_chunk(all_chunks, used_indices)
        if idx is None or chunk_text is None:
            print("You've answered all available questions!")
            break

        used_indices.add(idx)
        question_num += 1
        question, correct_ans = parse_chunk(chunk_text)

        print(f"Question {question_num}:")
        print(question)
        student_answer = input("\nYour answer (or 'exit'): ").strip()

        if student_answer.lower() == "exit":
            break

        prompt = eval_prompt.format(
            subject=subject_key,
            question=question,
            correct_answer=correct_ans,
            student_answer=student_answer,
        )
        evaluation = model.invoke(prompt).content
        score = parse_score(evaluation)
        scores.append(score)

        print("\nEvaluation:")
        print(evaluation)

        followup_chunk = get_followup_chunk(
            db=db,
            subject_path=subject_path,
            current_chunk_text=chunk_text,
            current_question=question,
            local_chunks=local_chunks,
            used_indices=used_indices,
        )

        if followup_chunk:
            fq, fa = parse_chunk(followup_chunk)
            print("\nFollow-up:")
            print(fq)
            followup_answer = input("\nYour answer (or 'exit'): ").strip()

            if followup_answer.lower() == "exit":
                break

            prompt = eval_prompt.format(
                subject=subject_key,
                question=fq,
                correct_answer=fa,
                student_answer=followup_answer,
            )
            evaluation = model.invoke(prompt).content
            score = parse_score(evaluation)
            scores.append(score)

            print("\nEvaluation:")
            print(evaluation)

        print()

    if scores:
        total_score = sum(scores)
        max_score = len(scores) * 10
        scaled_score = round((total_score / max_score) * 100)
        print("\n----SUMMARY----")
        print(f"  Subject            : {subject_key.title()}")
        print(f"  Questions answered : {len(scores)}")
        print(f"  Raw score          : {total_score}/{max_score}")
        print(f"  Final score        : {scaled_score}/100")
    else:
        print("\nNo questions answered. See you next time!\n")


if __name__ == "__main__":
    main()
