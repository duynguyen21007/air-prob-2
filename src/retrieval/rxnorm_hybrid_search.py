import csv
from pathlib import Path
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
import re

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
        
        # 4. Setup Ensemble (favoring Dense 80%)
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.chroma_retriever],
            weights=[0.2, 0.8]
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

    def get_qualified_rxcuis(self, query: str, margin: float = 0.05) -> list[str]:
        """
        Run the ensemble retriever (oversampled), deduplicate to top 20 unique RXCUIs,
        rerank them, and return all RXCUIs within the specified margin of the best score.
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
        qualified_rxcuis = []
        
        for score, doc in zip(scores, unique_candidates):
            if score >= top_score - margin:
                qualified_rxcuis.append(doc.metadata.get("rxcui"))
                
        return qualified_rxcuis
