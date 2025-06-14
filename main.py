import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai
from fastapi.concurrency import run_in_threadpool

# Load environment variables from .env file
load_dotenv()
api_key = os.getenv("KEY")
environment = os.getenv("ENV", "production")  # default is production

genai.configure(api_key=api_key)

# Enable docs only in development
app = FastAPI(
    title="Interview Question Generator API",
    docs_url="/docs" if environment == "development" else None,
    redoc_url="/redoc" if environment == "development" else None,
    openapi_url="/openapi.json" if environment == "development" else None,
)

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace * with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request body structure
class PromptRequest(BaseModel):
    role: str

@app.post("/generate")
async def generate_questions(request: PromptRequest):
    role = request.role

    prompt = f"""
Generate 10 interview questions for a {role}. 
Categorize them into easy, medium, and hard.
Format:
Easy:
- Q1
- Q2
- Q3
Medium:
- Q4
- Q5
- Q6
- Q7
Hard:
- Q8
- Q9
- Q10
    """
    try:
        model = genai.GenerativeModel("models/gemini-1.5-flash")
        response = await run_in_threadpool(model.generate_content, prompt)
        questions = getattr(response, "text", None) or getattr(response, "content", None)
        if not questions:
            raise HTTPException(status_code=500, detail="No questions generated.")
        return {"questions": questions}
    except Exception as e:
        raise HTTPException(status_code=429, detail="Quota exceeded or API error: " + str(e))
