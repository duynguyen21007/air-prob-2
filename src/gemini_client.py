import time
from google import genai
from google.genai import types
from src.config import GEMINI_API_KEY, MODEL_NAME, TEMPERATURE, MAX_RETRIES, RETRY_BASE_DELAY, RPM_LIMIT

# Initialize the Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)

_last_call_time = 0.0

def generate_structured_response(prompt: str, response_schema, system_instruction: str = None) -> any:
    """
    Calls the Gemini API and enforces a structured JSON output based on the provided Pydantic schema.
    Includes rate limiting and retry logic.
    """
    global _last_call_time
    
    config = types.GenerateContentConfig(
        temperature=TEMPERATURE,
        response_mime_type="application/json",
        response_schema=response_schema,
    )
    if system_instruction:
         config.system_instruction = system_instruction
         
    min_interval = 60.0 / RPM_LIMIT
    
    for attempt in range(MAX_RETRIES):
        now = time.time()
        elapsed = now - _last_call_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
            
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=config,
            )
            _last_call_time = time.time()
            return response.parsed
        except Exception as e:
            _last_call_time = time.time()
            error_str = str(e).lower()
            if "429" in error_str or "503" in error_str or "500" in error_str or "quota" in error_str:
                if attempt < MAX_RETRIES - 1:
                    sleep_time = RETRY_BASE_DELAY * (2 ** attempt)
                    print(f"API Rate limit or server error ({e}). Retrying in {sleep_time}s...")
                    time.sleep(sleep_time)
                else:
                    raise
            else:
                raise
