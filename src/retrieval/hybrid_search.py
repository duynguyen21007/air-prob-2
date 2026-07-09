import csv
from pathlib import Path
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from pyvi import ViTokenizer

def vietnamese_tokenizer(text: str) -> list[str]:
    return ViTokenizer.tokenize(text.lower()).split()

class HybridSearcher:
    def __init__(self, data_csv_path: str, chroma_persist_dir: str):
        """
        Initialize the HybridSearcher.
        It loads the ICD-10 data for BM25 and connects to the existing ChromaDB.
        """
        self.data_csv_path = Path(data_csv_path)
        self.chroma_persist_dir = Path(chroma_persist_dir)
        
        # 1. Load documents for BM25
        documents = self._load_documents()
        
        # 2. Setup BM25 Retriever (Sparse) - Top 5
        # It's fast enough to compute BM25 index on the fly.
        self.bm25_retriever = BM25Retriever.from_documents(
            documents,
            preprocess_func=vietnamese_tokenizer
        )
        self.bm25_retriever.k = 5
        
        # 3. Setup Chroma Retriever (Dense) - Top 5
        # The vector database must be pre-built by build_retrieval_index.py
        embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")
        self.vectorstore = Chroma(
            persist_directory=str(self.chroma_persist_dir),
            embedding_function=embeddings,
            collection_name="icd10_collection"
        )
        self.chroma_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
        
        # 4. Setup Ensemble (favoring Dense 80%)
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.chroma_retriever],
            weights=[0.2, 0.8]
        )

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
        results = self.ensemble_retriever.invoke(e5_query)
        
        if not results:
            return None
            
        # results are ranked by RRF score. The first one is the best.
        best_doc = results[0]
        return best_doc.metadata.get("icd")

    def get_top_k_icds(self, query: str, k: int = 5) -> list[str]:
        """
        Run the ensemble retriever and return the top K ICD-10 codes with their keywords.
        """
        e5_query = f"query: {query}"
        results = self.ensemble_retriever.invoke(e5_query)
        
        top_k = []
        for doc in results[:k]:
            icd = doc.metadata.get("icd")
            # The page_content has 'passage: ' prefix from the dense index, let's strip it for cleaner review
            content = doc.page_content.replace("passage: ", "")
            top_k.append(f"{icd} ({content})")
            
        return top_k
