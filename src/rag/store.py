import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

class RAGSystem:
    def __init__(self, data_path="data"):
        provider = os.getenv("EMBEDDING_PROVIDER", "openai").lower()
        model_name = os.getenv("EMBEDDING_MODEL") # LangChain defaults if None
        
        if provider == "openai":
            self.embeddings = OpenAIEmbeddings(model=model_name) if model_name else OpenAIEmbeddings()
        elif provider == "vertexai":
            project_id = os.getenv("PROJECT_ID")
            location = os.getenv("LOCATION", "us-central1")
            api_key = os.getenv("VERTEX_API_KEY")
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model=model_name if model_name else "text-embedding-005",
                project=project_id,
                location=location,
                api_key=api_key
            )
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")

        self.persist_directory = "db"
        self.data_path = data_path
        self.vectorstore = None

    def initialize(self):
        """
        Loads data from the data directory and initializes the vector store.
        If the vector store already exists on disk, it loads it.
        """
        if os.path.exists(self.persist_directory) and os.listdir(self.persist_directory):
            print("Loading existing vector store...")
            self.vectorstore = Chroma(persist_directory=self.persist_directory, embedding_function=self.embeddings)
        else:
            print("Creating new vector store...")
            self.vectorstore = self._create_vectorstore()

    def _create_vectorstore(self):
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
            # Create a dummy file if none exists to avoid errors
            with open(os.path.join(self.data_path, "knowledge.txt"), "w") as f:
                f.write("This is a sample knowledge base for the Slack bot.")

        documents = []
        for file in os.listdir(self.data_path):
            if file.endswith(".txt"):
                loader = TextLoader(os.path.join(self.data_path, file))
                documents.extend(loader.load())

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)
        
        return Chroma.from_documents(
            documents=docs, 
            embedding=self.embeddings, 
            persist_directory=self.persist_directory
        )

    def get_retriever(self):
        if not self.vectorstore:
            self.initialize()
        return self.vectorstore.as_retriever()


