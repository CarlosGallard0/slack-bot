import os
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from raptor_pipeline import RaptorBuilder, RaptorRetriever

PERSIST_DIR = "./.raptor_checkpoints"
COLLECTION = "medical_papers"

print("📂 Loading RAPTOR from:", PERSIST_DIR)

builder = RaptorBuilder.from_existing(
    collection_name=COLLECTION,
    persist_directory=PERSIST_DIR,
)

if builder is None:
    raise RuntimeError("❌ RAPTOR builder not found")

print(f"✅ RAPTOR tree loaded | nodes={len(builder.nodes)}")

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

retriever = RaptorRetriever(
    vectorstore=vectorstore,
    nodes=builder.nodes,
    tree_structure=builder.tree_structure,
    default_mode="tree_traversal",
)

print("\n🧠 Ask a question:")
question = input("> ").strip()

results = retriever.retrieve(question, top_k=5)

print("\n🔎 Results:\n")
for i, r in enumerate(results, 1):
    print(f"[{i}] Source: {r['source_doc']}")
    print(r["text"][:600])
    print("-" * 60)
