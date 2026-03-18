# RAG-based-QnA

## FastAPI Interview Evaluator

### 1. Build the vector DB

```powershell
python .\RAG\make_db.py
```

### 2. Run the API server

```powershell
uvicorn RAG.server:app --reload
```

### 3. Endpoints

- `GET /health`
- `POST /session/new`
- `GET /session/{session_id}/question`
- `POST /session/{session_id}/answer`
- `GET /session/{session_id}/summary`

### 4. Example flow

1. Create session:
```http
POST /session/new
```
Response:
```json
{
  "session_id": "uuid",
  "total_questions": 40
}
```

2. Get a question:
```http
GET /session/{session_id}/question
```

3. Submit answer:
```http
POST /session/{session_id}/answer
Content-Type: application/json

{
  "question_id": 12,
  "student_answer": "Your answer..."
}
```

Evaluation response schema:
```json
{
  "score": 0,
  "factuality": 0,
  "context": 2,
  "originality": 0,
  "example": 0,
  "injection": true,
  "feedback": "..."
}
```
