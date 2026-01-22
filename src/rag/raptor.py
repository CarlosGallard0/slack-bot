import os
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from raptor_pipeline import RaptorBuilder, RaptorRetriever


class RaptorClient:
    def __init__(self):
        COLLECTION = "medical_papers"
        PERSIST_DIR = os.getenv("RAPTOR_PERSIST_DIR", "/raptor_checkpoints")

        builder = RaptorBuilder.from_existing(
            collection_name=COLLECTION,
            persist_directory=PERSIST_DIR,
        )

        if builder is None:
            raise RuntimeError("RAPTOR not available")

        embeddings = GoogleGenerativeAIEmbeddings(
            model="text-embedding-005",
            vertexai=True,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )

        vectorstore = Chroma(
            collection_name=COLLECTION,
            embedding_function=embeddings,
            persist_directory=f"{PERSIST_DIR}/{COLLECTION}/chroma",
        )

        self.retriever = RaptorRetriever(
            vectorstore=vectorstore,
            nodes=builder.nodes,
            tree_structure=builder.tree_structure,
            default_mode="tree_traversal",
        )

    def search(self, query: str, top_k: int = 5):
        results = self.retriever.retrieve(query, top_k=top_k)

        return [
            {
                "text": r["text"],
                "source": r["source_doc"],
                "score": r["score"],
            }
            for r in results
        ]
