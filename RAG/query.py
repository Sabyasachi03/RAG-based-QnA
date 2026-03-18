import random
import re
import os
import warnings
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma
    warnings.filterwarnings(
        "ignore",
        message=r"The class `Chroma` was deprecated in LangChain 0\.2\.9.*",
    )

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv()
load_dotenv(os.path.join(BASE_DIR, ".env"))

CHROMA_PATH = os.path.join(BASE_DIR, "chroma")

EVAL_TEMPLATE = """
You are evaluating a student's answer to a machine learning question.

Question: {question}
Correct Answer: {correct_answer}
Student's Answer: {student_answer}

Give a short evaluation (2-3 sentences): was the student correct, what did they miss or get right?
Then on a new line write exactly: Score: X/10
where X is a number from 0 to 10.
"""

def parse_score(evaluation: str) -> int:
    """Extract the numeric score from the evaluation text."""
    match = re.search(r"Score:\s*(\d+)/10", evaluation)
    return int(match.group(1)) if match else 0

def get_random_chunk(all_chunks, used_indices):
    """Pick a random unused chunk index."""
    available = [i for i in range(len(all_chunks["documents"])) if i not in used_indices]
    if not available:
        return None, None
    idx = random.choice(available)
    return idx, all_chunks["documents"][idx]

def parse_chunk(chunk_text):
    """Split chunk into question and answer."""
    lines = chunk_text.split("\n", 1)
    question    = lines[0].replace("Q: ", "").strip()
    correct_ans = lines[1].replace("A: ", "").strip() if len(lines) > 1 else ""
    return question, correct_ans

def main():
    # Load DB and model
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Add it to RAG/.env or set it in your environment."
        )

    embedding_function = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key,
    )
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
    )
    eval_prompt = ChatPromptTemplate.from_template(EVAL_TEMPLATE)

    all_chunks = db.get()
    total_chunks = len(all_chunks["documents"])
    if total_chunks == 0:
        print("No chunks found in DB. Run make_db.py first.")
        return

    print("\n-----ML Quiz-----")
    print("Type 'exit' at any time to stop and see your score.")
    print("\n")

    used_indices  = set()
    scores        = []
    question_num  = 0

    while True:
        # Pick a random unused chunk
        idx, chunk_text = get_random_chunk(all_chunks, used_indices)
        if idx is None:
            print("You've answered all available questions!")
            break
        used_indices.add(idx)
        question_num += 1
        question, correct_ans = parse_chunk(chunk_text)

        # Ask the question
        print(f"Question {question_num}:")
        print(question)
        student_answer = input("\nYour answer (or 'exit'): ").strip()

        if student_answer.lower() == "exit":
            break

        # Evaluate
        prompt = eval_prompt.format(
            question=question,
            correct_answer=correct_ans,
            student_answer=student_answer,
        )
        evaluation = model.invoke(prompt).content
        score = parse_score(evaluation)
        scores.append(score)

        print("\nEvaluation:")
        print(evaluation)

        # Follow-up from DB
        followup_results = db.similarity_search_with_relevance_scores(question, k=4)
        followup_chunk = None
        for doc, _ in followup_results:
            if doc.page_content.strip() != chunk_text.strip():
                followup_chunk = doc.page_content
                break

        if followup_chunk:
            fq, fa = parse_chunk(followup_chunk)
            print("\nFollow-up:")
            print(fq)
            followup_answer = input("\nYour answer (or 'exit'): ").strip()

            if followup_answer.lower() == "exit":
                break

            # Evaluate follow-up
            prompt = eval_prompt.format(
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

    # Final score
    if scores:
        total_score  = sum(scores)
        max_score    = len(scores) * 10
        scaled_score = round((total_score / max_score) * 100)
        print("\n----SUMMARY----")
        print(f"  Questions answered : {len(scores)}")
        print(f"  Raw score          : {total_score}/{max_score}")
        print(f"  Final score        : {scaled_score}/100")
    else:
        print("\nNo questions answered. See you next time!\n")

if __name__ == "__main__":
    main()
