import os
from google.cloud import storage

from langchain_chroma import Chroma
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)

from raptor_pipeline import (
    RaptorBuilder,
    RaptorRetriever,
)

GCP_BUCKET = "raptor-checkpoints-neurodocsdomain"
GCP_PREFIX = "medical_papers/"
COLLECTION_NAME = "medical_papers"
PERSIST_DIR = "./.raptor_checkpoints"


assert os.getenv("GOOGLE_API_KEY"), "❌ GOOGLE_API_KEY is missing"


def download_raptor_from_gcp(
    bucket_name: str,
    gcs_prefix: str,
    local_dir: str,
):
    print("☁️ Downloading RAPTOR data from GCP...")

    client = storage.Client()

    blobs = client.list_blobs(bucket_name, prefix=gcs_prefix)

    count = 0
    for blob in blobs:
        if blob.name.endswith("/"):
            continue

        relative_path = blob.name[len(gcs_prefix) :]
        local_path = os.path.join(local_dir, relative_path)

        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        blob.download_to_filename(local_path)
        count += 1

    print(f"✅ Downloaded {count} files from GCP")


download_raptor_from_gcp(
    bucket_name=GCP_BUCKET,
    gcs_prefix=GCP_PREFIX,
    local_dir=f"{PERSIST_DIR}/{COLLECTION_NAME}",
)

print("🔄 Loading RAPTOR...")

builder = RaptorBuilder.from_existing(
    collection_name=COLLECTION_NAME,
    persist_directory=PERSIST_DIR,
)

nodes = builder.nodes
tree_structure = builder.tree_structure

print(f"✅ RAPTOR loaded | Nodes: {len(nodes)} | Levels: {len(tree_structure)}")


embeddings = GoogleGenerativeAIEmbeddings(
    model="text-embedding-005",
    vertexai=True,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=f"{PERSIST_DIR}/{COLLECTION_NAME}/chroma",
)

retriever = RaptorRetriever(
    vectorstore=vectorstore,
    nodes=nodes,
    tree_structure=tree_structure,
    default_mode="tree_traversal",
)

print("\n 🔍 Ask your question:")
question = input("> ").strip()

if not question:
    print("❌ No question provided")
    exit(1)


print("\n🔍 Retrieving relevant fragments...\n")

raw_results = retriever.retrieve(
    query=question,
    top_k=5,
    mode="tree_traversal",
)

if not raw_results:
    print("⚠️ No results found")
    exit(0)

review_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.0,
    vertexai=True,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

fragments = []
for i, r in enumerate(raw_results):
    fragments.append(
        f"Fragment {i+1}:\n"
        f"Source: {r['source_doc']}\n"
        f"Level: {r['level']}\n"
        f"Text:\n{r['text']}\n"
    )

prompt = f"""
You are a medical review agent.

Question:
{question}

Below are text fragments retrieved from a hierarchical retrieval system.
Your task:
- Select ONLY the fragments that directly answer the question.
- Discard unrelated information.
- Prefer original study findings.
- Use ONLY the provided fragments.
- Do NOT add external knowledge.

Fragments:
{chr(10).join(fragments)}
"""

print("🧠 Generating final answer...\n")

response = review_llm.invoke(prompt)

print("======================================")
print("✅ FINAL ANSWER\n")
print(response.content)
