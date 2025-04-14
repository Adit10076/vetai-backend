from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import json
import os
from typing import List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# Get frontend URL from environment variable, default to localhost:3000 if not set
frontend_url = os.getenv("FRONTEND_URL", "http://localhost:3000")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_url],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class StartupIdea(BaseModel):
    title: str
    problem: str
    solution: str
    audience: str
    businessModel: str

class Score(BaseModel):
    overall: float
    marketPotential: float
    technicalFeasibility: float

class SwotAnalysis(BaseModel):
    strengths: List[str]
    weaknesses: List[str]
    opportunities: List[str]
    threats: List[str]

class MarketAnalysis(BaseModel):
    targetMarket: str
    tam: str
    sam: str
    som: str
    growthRate: str
    trends: List[str]
    competitors: List[str]
    customerNeeds: List[str]
    barriersToEntry: List[str]

class StartupEvaluation(BaseModel):
    score: Score
    swotAnalysis: SwotAnalysis
    mvpSuggestions: List[str]
    businessModelIdeas: List[str]
    marketAnalysis: MarketAnalysis

FALLBACK_RESPONSE = {
    "score": {"overall": 75, "marketPotential": 70, "technicalFeasibility": 80},
    "swotAnalysis": {
        "strengths": ["Innovative", "Scalable", "Well-targeted"],
        "weaknesses": ["High dev cost", "Low adoption risk", "Unclear pricing"],
        "opportunities": ["Growing market", "Tech trends", "Global reach"],
        "threats": ["Regulations", "Competitors", "Economic instability"]
    },
    "mvpSuggestions": ["Build landing page", "Create waitlist", "Offer demo"],
    "businessModelIdeas": ["Subscription", "Freemium", "Tiered pricing"],
    "marketAnalysis": {
        "targetMarket": "Urban eco-conscious youth",
        "tam": "$50000000000",
        "sam": "$5000000000",
        "som": "$100000000",
        "growthRate": "15% CAGR due to rising demand for sustainable consumer products globally",
        "trends": ["AI for sustainability", "Eco-lifestyle tracking"],
        "competitors": ["Greenly", "Joro"],
        "customerNeeds": ["Actionable tips", "Progress tracking"],
        "barriersToEntry": ["Trust", "Accuracy", "Engagement"]
    }
}

@app.post("/validate", response_model=StartupEvaluation)
async def validate_startup_idea(idea: StartupIdea):
    prompt = f"""
You are a startup evaluator. Analyze the following startup idea and return valid JSON only. Do not repeat input.

Startup:
Title: {idea.title}
Problem: {idea.problem}
Solution: {idea.solution}
Audience: {idea.audience}
Business Model: {idea.businessModel}

Output JSON format:
{{
  "isGibberish": boolean,
  "score": {{
    "overall": number [0-100],
    "marketPotential": number [0-100],
    "technicalFeasibility": number [0-100]
  }},
  "swotAnalysis": {{
    "strengths": [string, ...],
    "weaknesses": [string, ...],
    "opportunities": [string, ...],
    "threats": [string, ...]
  }},
  "mvpSuggestions": [string, string, string],
  "businessModelIdeas": [string, ...],
  "marketAnalysis": {{
    "targetMarket": string,
    "tam": string (total addressable market in USD, numeric format only, e.g. "$1500000000", and mention the user types or groups included in TAM),
    "sam": string (serviceable available market in USD, and mention who is actually reachable based on your scope),
    "som": string (serviceable obtainable market in USD, and mention who is most likely to convert first),
    "growthRate": string (state the CAGR or growth and the reason behind this growth based on market forces or user demand),
    "trends": [string, ...],
    "competitors": [string, ...],
    "customerNeeds": [string, ...],
    "barriersToEntry": [string, ...]
  }}
}}
- Return only valid JSON
- Do not repeat or rephrase the input.
- No markdown, no commentary.
"""

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            check = await client.get("http://localhost:11434/api/tags")
            if check.status_code != 200:
                raise HTTPException(status_code=503, detail="Ollama not available")

        text_response = ""
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream("POST", "http://localhost:11434/api/generate",
                                     json={"model": "mistral", "prompt": prompt, "stream": True}) as response:
                async for line in response.aiter_lines():
                    if line.strip():
                        try:
                            piece = json.loads(line)
                            text_response += piece.get("response", "")
                        except:
                            continue

        if text_response.strip().startswith("{") and text_response.strip().endswith("}"):
            return json.loads(text_response)

        json_start = text_response.find("{")
        json_end = text_response.rfind("}") + 1
        json_str = text_response[json_start:json_end]
        result = json.loads(json_str)

        if all(k in result for k in ["score", "swotAnalysis", "mvpSuggestions", "businessModelIdeas", "marketAnalysis"]):
            return result
        return FALLBACK_RESPONSE

    except Exception as e:
        print("Streaming or parsing error:", str(e))
        raise HTTPException(status_code=500, detail="Error generating response")