import csv
import json
from pathlib import Path
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from pyvi import ViTokenizer
from sentence_transformers import CrossEncoder
import torch

from src.llm_client import get_response_for_single_chat

device = "cuda" if torch.cuda.is_available() else "cpu"


prompt_template = """You are an expert in medical ICD-10 coding.

# Task
Given a clinical text and a list of candidate ICD-10 codes, select all ICD-10 codes that are explicitly supported by the text.

# Rules
- Only choose codes from the provided candidate list.
- Return a JSON list containing only the selected ICD-10 code strings.
- If no candidate matches the text, return [].
- If the text is meaningless, garbage, or not related to a patient's diagnosis or treatment, return [].
- Do not infer diagnoses that are not explicitly mentioned.
- Do not return explanations or any additional text.

# Example

Text:
"Migrain tiền đình / Mất ngủ"

Candidates:
G43.9: Bệnh đau nửa đầu [migraine], không xác định
J00: Nhiễm trùng đường hô hấp trên cấp tính
D04: Ung thư biểu mô tại chỗ ở da

Output:
[
  "G43.9"
]

---

# Input

Text:
"{text}"

Candidates:
{candidates}

# Output
"""


def _postprocess_icd_response(s, icds):
    """Parse LLM response to extract ICD-10 codes, validating against known set."""
    if not isinstance(s, str):
        return []
    if ']' not in s:
        return []
    if '[' not in s:
        return []
    s = s[s.find('['): s.find(']') + 1]
    try:
        s = json.loads(s)
        if not isinstance(s, list):
            return []
        for x in s:
            if not isinstance(x, str):
                return []
            if x not in icds:
                return []
        return s
    except:
        return []


def vietnamese_tokenizer(text: str) -> list[str]:
    return ViTokenizer.tokenize(text.lower()).split()


class Icd10HybridSearcher:
    def __init__(self, data_csv_path: str, chroma_persist_dir: str):
        """
        Initialize the Icd10HybridSearcher.
        It loads the ICD-10 data for BM25 and connects to the existing ChromaDB.
        """
        self.data_csv_path = Path(data_csv_path)
        self.chroma_persist_dir = Path(chroma_persist_dir)
        
        # 1. Load documents for BM25
        documents = self._load_documents()
        
        # 2. Setup BM25 Retriever (Sparse) - oversample before reranking
        self.bm25_retriever = BM25Retriever.from_documents(
            documents,
            preprocess_func=vietnamese_tokenizer
        )
        self.bm25_retriever.k = 150
        
        # 3. Setup Chroma Retriever (Dense) - oversample before reranking
        embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")
        self.vectorstore = Chroma(
            persist_directory=str(self.chroma_persist_dir),
            embedding_function=embeddings,
            collection_name="icd10_collection"
        )
        self.chroma_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 150})
        
        # 4. Setup Ensemble (favoring Dense 80%)
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.chroma_retriever],
            weights=[0.2, 0.8]
        )
        
        # 5. Setup CrossEncoder Reranker
        self.reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512, device=device)

    def _load_documents(self) -> list[Document]:
        """Load ICD-10 data into LangChain Documents."""
        docs = []
        with open(self.data_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                icd_code = row.get("icd", "").strip()
                keyword = row.get("kw", "").strip()
                if icd_code and keyword:
                    docs.append(Document(page_content=f"passage: {keyword}", metadata={"icd": icd_code}))
        return docs

    def get_best_icd(self, query: str) -> str | None:
        """
        Run the ensemble retriever and return the best single ICD-10 code.
        """
        e5_query = f"query: {query}"
        candidates = self.ensemble_retriever.invoke(e5_query)
        
        if not candidates:
            return None
            
        pairs = [[query, doc.page_content.replace("passage: ", "")] for doc in candidates]
        scores = self.reranker.predict(pairs)
        
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        best_doc = candidates[best_idx]
        return best_doc.metadata.get("icd")

    def get_qualified_icds(
        self,
        query: str,
        margin: float = 0.05,
        absolute_threshold: float = 0.5,
        max_candidates: int = 5,
        include_content: bool = False,
    ) -> list[str]:
        """Return ICD-10 codes that pass absolute and relative reranker cutoffs."""
        e5_query = f"query: {query}"
        candidates = self.ensemble_retriever.invoke(e5_query)

        if not candidates:
            return []

        unique_candidates = []
        seen_icds = set()
        for doc in candidates:
            icd = doc.metadata.get("icd")
            if not icd or icd in seen_icds:
                continue
            seen_icds.add(icd)
            unique_candidates.append(doc)
            if len(unique_candidates) == 20:
                break

        if not unique_candidates:
            return []

        pairs = [
            [query, doc.page_content.replace("passage: ", "")]
            for doc in unique_candidates
        ]
        scores = self.reranker.predict(pairs)
        top_score = max(scores)

        if top_score < absolute_threshold:
            return []

        qualified = []
        scored_candidates = sorted(
            zip(scores, unique_candidates), key=lambda item: item[0], reverse=True
        )
        for score, doc in scored_candidates:
            if score < top_score - margin:
                break
            icd = doc.metadata["icd"]
            if include_content:
                content = doc.page_content.replace("passage: ", "")
                qualified.append(f"{icd} ({content})")
            else:
                qualified.append(icd)
            if len(qualified) >= max_candidates:
                break

        return qualified

    def get_qualified_icds_v2(
        self,
        query: str,
        max_candidates: int = 5,
    ) -> list[str]:
        """Return ICD-10 codes using LLM reranking on top ensemble candidates."""
        e5_query = f"query: {query}"
        candidates = self.ensemble_retriever.invoke(e5_query)

        if not candidates:
            return []

        unique_candidates = []
        seen_icds = set()
        for doc in candidates:
            icd = doc.metadata.get("icd")
            if not icd or icd in seen_icds:
                continue
            seen_icds.add(icd)
            unique_candidates.append(doc)
            if len(unique_candidates) == 20:
                break

        if not unique_candidates:
            return []

        candidate_strs = []
        for doc in unique_candidates:
            if doc.page_content.startswith("passage: "):
                doc.page_content = doc.page_content[len("passage: "):]
            icd = doc.metadata.get("icd")
            candidate_strs.append(f'{icd}: {doc.page_content}')

        prompt = prompt_template.format(text=query, candidates="\n".join(candidate_strs))
        response = get_response_for_single_chat(prompt)
        response = _postprocess_icd_response(response, seen_icds)
        return response

    def get_top_k_icds(self, query: str, k: int = 5) -> list[str]:
        """
        Run the ensemble retriever and return the top K ICD-10 codes with their keywords.
        """
        e5_query = f"query: {query}"
        candidates = self.ensemble_retriever.invoke(e5_query)
        
        if not candidates:
            return []
            
        pairs = [[query, doc.page_content.replace("passage: ", "")] for doc in candidates]
        scores = self.reranker.predict(pairs)
        
        scored_candidates = list(zip(scores, candidates))
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        
        top_k = []
        for score, doc in scored_candidates[:k]:
            icd = doc.metadata.get("icd")
            content = doc.page_content.replace("passage: ", "")
            top_k.append(f"{icd} ({content})")
            
        return top_k
