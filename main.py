from fastapi import FastAPI, HTTPException
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

FALLBACK_RESPONSE = StartupEvaluation(
    score=Score(overall=75, marketPotential=70, technicalFeasibility=80),
    swotAnalysis=SwotAnalysis(
        strengths=["Strong concept", "Clear target market"],
        weaknesses=["High competition", "Complex implementation"],
        opportunities=["Market growth", "Tech advancements"],
        threats=["Regulatory changes", "Economic shifts"]
    ),
    mvpSuggestions=["Develop core feature", "Create landing page", "Conduct user testing"],
    businessModelIdeas=["Subscription model", "Freemium approach"],
    marketAnalysis=MarketAnalysis(
        targetMarket="Global tech users",
        tam="$50B",
        sam="$10B",
        som="$1B",
        growthRate="12% CAGR",
        trends=["Digital transformation", "AI adoption"],
        competitors=["Existing solutions", "Tech giants"],
        customerNeeds=["Ease of use", "Cost efficiency"],
        barriersToEntry=["Market saturation", "High capital needs"]
    )
)

def repair_json(json_str: str) -> Optional[dict]:
    """Attempt to fix common JSON formatting issues"""
    try:
        # First pass cleanup
        clean_str = json_str.strip()
        
        # Remove JSON markdown blocks
        if clean_str.startswith("```json"):
            clean_str = clean_str[7:]
        if clean_str.endswith("```"):
            clean_str = clean_str[:-3].strip()
            
        # Fix common escaping issues
        clean_str = clean_str.replace("\\n", "") \
                            .replace("\\'", "'") \
                            .replace('\\"', '"') \
                            .replace("True", "true") \
                            .replace("False", "false")
                            
        # Handle trailing commas
        clean_str = clean_str.replace(",}", "}").replace(",]", "]")
        
        return json.loads(clean_str)
    except json.JSONDecodeError as e:
        logger.error(f"JSON repair failed: {str(e)}")
        return None

@app.post("/validate", response_model=StartupEvaluation)
async def validate_startup_idea(idea: StartupIdea):
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if not GROQ_API_KEY:
        logger.error("Missing GROQ_API_KEY environment variable")
        return FALLBACK_RESPONSE

    prompt = f"""
    You are a highly experienced startup analyst trusted by top-tier VCs. Your role is to critically and realistically evaluate the following startup idea and return your findings as structured JSON.
    
    Startup Details:
    Title: {idea.title}
    Problem: {idea.problem}
    Solution: {idea.solution}
    Target Audience: {idea.audience}
    Business Model: {idea.businessModel}
    
    Evaluation Guidelines:
    1. First, assess if the input is incoherent, vague, or gibberish. If so, set `"isGibberish": true` and return generic, low-confidence responses. Otherwise, set `"isGibberish": false`.
    2. Provide realistic, well-justified scores (0–100) for:
       - overall potential
       - market potential
       - technical feasibility  
       Avoid generic or inflated scoring. Use critical thinking.
    3. Perform a thoughtful SWOT analysis (3–5 points each) tailored to the startup's domain. Avoid clichés or shallow insights.
    4. Suggest 3 realistic MVP features or actions to validate the idea early.
    5. Propose 2–3 feasible business model options that align with the problem and audience.
    6. Conduct a market analysis including:
       - clear definition of the target market
       - realistic estimates for TAM/SAM/SOM (with dollar figures or user counts)
       - growth rate and trends based on common knowledge or patterns
       - likely competitors (real or credible fictional names)
       - key customer needs
       - barriers to entry (tech, regulation, cost, trust, etc.)
    
    Your output must be the best possible analysis based on the information given. Make assumptions if necessary, but keep them realistic.
    
     Output only **valid minified JSON**. No markdown, comments, explanations, or extra text.
    
    Required Output Format:
    {{
      "isGibberish": boolean,
      "score": {{
        "overall": number,
        "marketPotential": number,
        "technicalFeasibility": number
      }},
      "swotAnalysis": {{
        "strengths": [string],
        "weaknesses": [string],
        "opportunities": [string],
        "threats": [string]
      }},
      "mvpSuggestions": [string],
      "businessModelIdeas": [string],
      "marketAnalysis": {{
        "targetMarket": string,
        "tam": string,
        "sam": string,
        "som": string,
        "growthRate": string,
        "trends": [string],
        "competitors": [string],
        "customerNeeds": [string],
        "barriersToEntry": [string]
      }}
    }}
    """


    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    payload = {
        "model": "llama3-70b-8192",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 2500,
        "response_format": {"type": "json_object"},
        "top_p": 0.9,
        "frequency_penalty": 0.5
    }

    try:
        async with httpx.AsyncClient(timeout=40.0) as client:
            # Test API connection
            models_response = await client.get(
                "https://api.groq.com/openai/v1/models",
                headers=headers
            )
            if models_response.status_code != 200:
                logger.error(f"API connection failed: {models_response.text}")
                return FALLBACK_RESPONSE

            # Main request
            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            
            response_data = response.json()
            if not response_data.get("choices"):
                logger.error("Empty choices array in response")
                return FALLBACK_RESPONSE
                
            llm_content = response_data["choices"][0]["message"]["content"]
            logger.debug(f"Raw LLM output:\n{llm_content}")
            
            # Attempt JSON repair
            parsed_data = repair_json(llm_content)
            if not parsed_data:
                logger.error("JSON repair failed. Original response:")
                logger.error(llm_content)
                return FALLBACK_RESPONSE
                
            try:
                validated = StartupEvaluation(**parsed_data)
                return validated
            except ValidationError as e:
                logger.error(f"Validation failed: {e.errors()}")
                logger.error(f"Invalid data: {parsed_data}")
                return FALLBACK_RESPONSE

    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error {e.response.status_code}: {e.response.text}")
        return FALLBACK_RESPONSE
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return FALLBACK_RESPONSE

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
