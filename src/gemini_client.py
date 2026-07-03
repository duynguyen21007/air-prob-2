from google import genai
from google.genai import types
from src.config import GEMINI_API_KEY, MODEL_NAME, TEMPERATURE

# Initialize the Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)

def generate_structured_response(prompt: str, response_schema, system_instruction: str = None) -> any:
    """
    Calls the Gemini API and enforces a structured JSON output based on the provided Pydantic schema.
    """
    config = types.GenerateContentConfig(
        temperature=TEMPERATURE,
        response_mime_type="application/json",
        response_schema=response_schema,
    )
    if system_instruction:
         config.system_instruction = system_instruction
         
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=config,
    )
    
    return response.parsed
