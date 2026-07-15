import re
import json
from src.local_llm_client import LocalLLM

class LLMReranker:
    def __init__(self):
        self.llm = LocalLLM.get_instance()

    def select_best_candidates(self, query: str, candidates: list[tuple[str, str]]) -> list[str]:
        """
        Takes a query and a list of (code, description) tuples.
        Returns a list of validated codes selected by the LLM.
        """
        if not candidates:
            return []
            
        prompt = f"You are a medical expert.\nEntity: {query}\nCandidates:\n"
        valid_codes = set()
        for code, description in candidates:
            prompt += f"{code}: {description}\n"
            valid_codes.add(code)
            
        prompt += "\nSelect 1 to 5 of the most accurate codes for the entity. Return ONLY a valid JSON array of strings containing the codes. Example: [\"CODE1\", \"CODE2\"]"
        
        try:
            response_text = self.llm.generate_response(prompt)
            
            # Use regex to find the JSON array in case the model adds extra text
            match = re.search(r'\[(.*?)\]', response_text, re.DOTALL)
            if not match:
                print(f"Warning: Could not parse JSON array from LLM response: {response_text}")
                return [candidates[0][0]]
                
            json_str = "[" + match.group(1) + "]"
            parsed_codes = json.loads(json_str)
            
            if not isinstance(parsed_codes, list):
                return [candidates[0][0]]
                
            # Strict Filtering
            filtered_codes = []
            for code in parsed_codes:
                if str(code) in valid_codes:
                    filtered_codes.append(str(code))
                    
            if not filtered_codes:
                return [candidates[0][0]]
                
            return filtered_codes[:5]
            
        except Exception as e:
            print(f"Error during LLM reranking: {e}")
            return [candidates[0][0]]
