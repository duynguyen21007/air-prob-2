import csv
from pathlib import Path
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
import re
import json

from openai import OpenAI
import time

OPENAI_API_KEY="dummy"
client = OpenAI(api_key=OPENAI_API_KEY, base_url="http://localhost:8211/v1")
llm_config = {
    #"model": "Qwen/Qwen3.6-27B-FP8",
    "model": "Qwen/Qwen3.5-9B",
    "max_token": 4096,
    "temperature": 0.0,
}


prompt_template = """You are an expert in medical RxNorm coding.

# Task
Given a clinical text and a list of candidate RxNorm concepts, select all RxNorm concepts that are explicitly mentioned or clearly supported by the text.

# Rules
- Only choose concepts from the provided candidate list.
- Return a JSON list containing only the selected RxNorm concept IDs (RxCUI) as strings.
- If no candidate matches the text, return [].
- If the text is meaningless, garbage, or not related to a patient's medication, prescription, or treatment, return [].
- Do not infer medications that are not explicitly mentioned.
- Match the medication mentioned in the text with the most appropriate candidate concept.
- Ignore dosage, strength, frequency, route, and formulation unless they are required to distinguish between candidate concepts.
- Do not return explanations or any additional text.

# Example

Text:
"guaifenesin ml po q6h:prn"

Candidates:
392085: Guaifenesin 800 mg oral tablet
313782: acetaminophen 325 mg ORAL TABLET 
1099279: Foster & Thrive Stool Softener 100mg Tablet

Output:
[
  "392085"
]

---

# Input

Text:
"{text}"

Candidates:
{candidates}

# Output
"""

def get_response_for_single_chat(prompt):
    start = time.time()
    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    try:
        response = client.chat.completions.create(
            model=llm_config['model'],
            temperature=llm_config['temperature'],
            max_tokens=llm_config['max_token'],
            messages = messages,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}}
            # reasoning_effort="low"
        )
        # print(response)
        response = response.choices[0].message.content
    except Exception as e:
        raise Exception(e)
    return response

def postprocess(s, acceptable):
    if not isinstance(s, str):
        return []
    if ']' not in s:
        return []
    if '[' not in s:
        return []
    s = s[s.find('['): s.find(']') + 1].strip()
    try:
        s = json.loads(s)
        print('json loads' , s, type(s))
        if not isinstance(s, list):
            return []
        for x in s:
            if not isinstance(x, str):
                return []
            if x not in acceptable:
                return []
        return s
    except:
        return []


def preprocess_rxnorm_text(text: str) -> str:
    text = text.lower()
    text = "".join([c if c.isalpha() or c.isdigit() else " " for c in text])
    # Insert space between numbers and letters
    text = re.sub(r'(\d)([a-z]+)', r'\1 \2', text)
    text = re.sub(r'([a-z]+)(\d)', r'\1 \2', text)
    # Normalize spaces
    text = " ".join(text.split())
    return text

def rxnorm_bm25_tokenizer(text: str) -> list[str]:
    return preprocess_rxnorm_text(text).split()

class RxNormHybridSearcher:
    def __init__(self, data_csv_path: str, chroma_persist_dir: str):
        """
        Initialize the RxNormHybridSearcher.
        It loads the RxNorm data for BM25 and connects to the existing ChromaDB.
        """
        self.data_csv_path = Path(data_csv_path)
        self.chroma_persist_dir = Path(chroma_persist_dir)
        
        # 1. Load documents for BM25
        documents = self._load_documents()
        
        # 2. Setup BM25 Retriever (Sparse) - Top 150
        self.bm25_retriever = BM25Retriever.from_documents(
            documents,
            preprocess_func=rxnorm_bm25_tokenizer
        )
        self.bm25_retriever.k = 150
        
        # 3. Setup Chroma Retriever (Dense) - Top 150
        embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")
        self.vectorstore = Chroma(
            persist_directory=str(self.chroma_persist_dir),
            embedding_function=embeddings,
            collection_name="rxnorm_collection"
        )
        self.chroma_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 150})
        
        # 4. Setup Ensemble (balanced)
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.chroma_retriever],
            weights=[0.5, 0.5]
        )
        
        # 5. Setup CrossEncoder Reranker
        self.reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512)

    def _load_documents(self) -> list[Document]:
        """Load RxNorm data into LangChain Documents."""
        docs = []
        with open(self.data_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # The CSV has 'RXCUI' and 'STR' columns
                rxcui = row.get("RXCUI", "").strip()
                keyword = row.get("STR", "").strip()
                if rxcui and keyword:
                    keyword = preprocess_rxnorm_text(keyword)
                    # MUST match the exact string used in Chroma so RRF fusion works!
                    docs.append(Document(page_content=f"passage: {keyword}", metadata={"rxcui": rxcui}))
        return docs

    def get_best_rxcui(self, query: str) -> str | None:
        """
        Run the ensemble retriever and return the best single RxNorm CUI.
        """
        clean_query = preprocess_rxnorm_text(query)
        # For e5 models, it's strictly required to prepend "query: " for queries.
        e5_query = f"query: {clean_query}"
        results = self.ensemble_retriever.invoke(e5_query)
        
        if not results:
            return None
            
        # results are ranked by RRF score. The first one is the best.
        best_doc = results[0]
        return best_doc.metadata.get("rxcui")

    def get_top_k_rxcuis(self, query: str, k: int = 5) -> list[str]:
        """
        Run the ensemble retriever and return the top K RxNorm CUIs with their keywords.
        """
        clean_query = preprocess_rxnorm_text(query)
        e5_query = f"query: {clean_query}"
        results = self.ensemble_retriever.invoke(e5_query)
        
        top_k = []
        for doc in results[:k]:
            rxcui = doc.metadata.get("rxcui")
            # The page_content has 'passage: ' prefix from the dense index, let's strip it for cleaner review
            content = doc.page_content.replace("passage: ", "")
            top_k.append(f"{rxcui} ({content})")
            
        return top_k

    def get_qualified_rxcuis(self, query: str, margin: float = 0.05, absolute_threshold: float = 0.0, max_candidates: int = 5, include_content: bool = False) -> list[str]:
        """
        Run the ensemble retriever (oversampled), deduplicate to top 20 unique RXCUIs,
        rerank them, and return all RXCUIs within the specified margin of the best score 
        AND above the absolute threshold.
        """
        clean_query = preprocess_rxnorm_text(query)
        e5_query = f"query: {clean_query}"
        candidates = self.ensemble_retriever.invoke(e5_query)
        
        if not candidates:
            return []
            
        unique_candidates = []
        seen_rxcuis = set()
        
        for doc in candidates:
            rxcui = doc.metadata.get("rxcui")
            if rxcui not in seen_rxcuis:
                seen_rxcuis.add(rxcui)
                unique_candidates.append(doc)
            if len(unique_candidates) == 20:
                break
                
        if not unique_candidates:
            return []
            
        pairs = [[query, doc.page_content.replace("passage: ", "")] for doc in unique_candidates]
        scores = self.reranker.predict(pairs)
        
        top_score = max(scores)
        
        # If the best score doesn't pass the absolute threshold, return 0 candidates
        if top_score < absolute_threshold:
            return []
            
        qualified_rxcuis = []
        
        # Sort candidates by score descending so we get the best ones first
        scored_candidates = sorted(zip(scores, unique_candidates), key=lambda x: x[0], reverse=True)
        
        for score, doc in scored_candidates:
            if score >= top_score - margin:
                rxcui = doc.metadata.get("rxcui")
                if include_content:
                    content = doc.page_content.replace("passage: ", "")
                    qualified_rxcuis.append(f"{rxcui} ({content})")
                else:
                    qualified_rxcuis.append(rxcui)
                    
            # Stop if we hit the maximum number of candidates
            if len(qualified_rxcuis) >= max_candidates:
                break
                
        return qualified_rxcuis



    def get_qualified_rxcuis_v2(self, query: str, max_candidates: int = 5) -> list[str]:
        """Return ICD-10 codes that pass absolute and relative reranker cutoffs."""
        clean_query = preprocess_rxnorm_text(query)
        e5_query = f"query: {clean_query}"
        candidates = self.ensemble_retriever.invoke(e5_query)
        if not candidates:
            return []

        unique_candidates = []
        seen_rxcuis = set()

        for doc in candidates:
            rxcui = doc.metadata.get("rxcui")
            rxcui = str(rxcui)
            if rxcui not in seen_rxcuis:
                seen_rxcuis.add(rxcui)
                unique_candidates.append(doc)
            if len(unique_candidates) == 20:
                break

        if not unique_candidates:
            return []

        candidates = []
        for doc in unique_candidates:
            if doc.page_content.startswith("passage: "):
                doc.page_content = doc.page_content[len("passage: "):]
            rxcui = doc.metadata.get("rxcui")
            candidates.append(f'{rxcui}: {doc.page_content}')

        prompt = prompt_template.format(text=query, candidates="\n".join(candidates))
        print('prompt', prompt)
        response = get_response_for_single_chat(prompt)
        print('response', response, type(response))
        response = postprocess(response, seen_rxcuis)
        print('seen_rxcuis', seen_rxcuis)
        print('response_list', response)
        return response
