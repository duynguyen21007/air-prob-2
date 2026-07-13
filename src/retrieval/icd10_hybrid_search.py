import csv
from pathlib import Path
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from pyvi import ViTokenizer
from sentence_transformers import CrossEncoder

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
        # It's fast enough to compute BM25 index on the fly.
        self.bm25_retriever = BM25Retriever.from_documents(
            documents,
            preprocess_func=vietnamese_tokenizer
        )
        self.bm25_retriever.k = 150
        
        # 3. Setup Chroma Retriever (Dense) - oversample before reranking
        # The vector database must be pre-built by build_icd10_index.py
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
        self.reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512)

    def _load_documents(self) -> list[Document]:
        """Load ICD-10 data into LangChain Documents."""
        docs = []
        with open(self.data_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # The CSV has 'icd' and 'kw' columns
                icd_code = row.get("icd", "").strip()
                keyword = row.get("kw", "").strip()
                if icd_code and keyword:
                    # MUST match the exact string used in Chroma to RRF fusion works!
                    docs.append(Document(page_content=f"passage: {keyword}", metadata={"icd": icd_code}))
        return docs

    def get_best_icd(self, query: str) -> str | None:
        """
        Run the ensemble retriever and return the best single ICD-10 code.
        """
        # For e5 models, it's strictly required to prepend "query: " for queries.
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

        # Keep one representative per code before reranking, so aliases do not
        # crowd out distinct ICD-10 alternatives.
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
            # The page_content has 'passage: ' prefix from the dense index, let's strip it for cleaner review
            content = doc.page_content.replace("passage: ", "")
            top_k.append(f"{icd} ({content})")
            
        return top_k
