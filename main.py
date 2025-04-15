from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError
from typing import List, Optional
import httpx
import json
import os
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = FastAPI()

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("FRONTEND_URL", "http://localhost:3000").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Models
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
        "strengths": ["Innovative solution", "Strong market need"],
        "weaknesses": ["High competition", "Technical complexity"],
        "opportunities": ["Market growth", "Emerging technologies"],
        "threats": ["Regulatory changes", "Economic downturn"]
    },
    "mvpSuggestions": ["Build prototype", "Conduct user testing"],
    "businessModelIdeas": ["Subscription model", "Freemium approach"],
    "marketAnalysis": {
        "targetMarket": "Global tech consumers",
        "tam": "$50B",
        "sam": "$10B",
        "som": "$1B",
        "growthRate": "10% CAGR",
        "trends": ["Digital transformation", "Remote work"],
        "competitors": ["Existing solutions", "Tech giants"],
        "customerNeeds": ["Ease of use", "Cost-effectiveness"],
        "barriersToEntry": ["Market saturation", "High capital needs"]
    }
}

def validate_and_clean_response(raw_response: str) -> Optional[dict]:
    """Clean and validate the LLM response"""
    try:
        # Remove JSON markdown blocks
        clean_json = raw_response.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(clean_json)
        
        # Basic validation
        if "score" not in parsed or "swotAnalysis" not in parsed:
            raise ValueError("Missing required fields in response")
            
        return parsed
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Response validation failed: {str(e)}")
        logger.debug(f"Invalid response content: {raw_response}")
        return None

@app.post("/validate", response_model=StartupEvaluation)
async def validate_startup_idea(idea: StartupIdea):
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if not GROQ_API_KEY:
        logger.error("Groq API key not found in environment variables")
        return FALLBACK_RESPONSE

    prompt = f"""
As a senior startup analyst, thoroughly evaluate this business idea. Provide detailed analysis in JSON format.

Startup Details:
Title: {idea.title}
Problem: {idea.problem}
Solution: {idea.solution}
Target Audience: {idea.audience}
Business Model: {idea.businessModel}

Analysis Requirements:
1. Score the idea (0-100) on overall potential, market potential, and technical feasibility
2. SWOT analysis with 3-5 points per category
3. 3 MVP suggestions
4. 2-3 business model ideas
5. Detailed market analysis including TAM/SAM/SOM

Output must be valid JSON only. Structure:
{{
  "score": {{
    "overall": number,
    "marketPotential": number,
    "technicalFeasibility": number
  }},
  "swotAnalysis": {{
    "strengths": [],
    "weaknesses": [],
    "opportunities": [],
    "threats": []
  }},
  "mvpSuggestions": [],
  "businessModelIdeas": [],
  "marketAnalysis": {{
    "targetMarket": string,
    "tam": string,
    "sam": string,
    "som": string,
    "growthRate": string,
    "trends": [],
    "competitors": [],
    "customerNeeds": [],
    "barriersToEntry": []
  }}
}}
"""

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    payload = {
        "model": "mixtral-8x7b-32768",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 2000,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload
            )
            
            # Check for HTTP errors
            response.raise_for_status()
            
            response_data = response.json()
            logger.debug(f"API Response: {json.dumps(response_data, indent=2)}")
            
            if not response_data.get("choices"):
                logger.error("Empty choices array in response")
                return FALLBACK_RESPONSE
                
            llm_response = response_data["choices"][0]["message"]["content"]
            logger.info(f"Raw LLM response: {llm_response}")
            
            parsed_data = validate_and_clean_response(llm_response)
            if not parsed_data:
                logger.warning("Using fallback response due to invalid LLM output")
                return FALLBACK_RESPONSE
                
            # Validate against Pydantic model
            validated_response = StartupEvaluation(**parsed_data)
            return validated_response
            
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error {e.response.status_code}: {e.response.text}")
        return FALLBACK_RESPONSE
        
    except ValidationError as e:
        logger.error(f"Response validation failed: {str(e)}")
        return FALLBACK_RESPONSE
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return FALLBACK_RESPONSE

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
