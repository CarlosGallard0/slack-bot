import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from src.rag.store import RAGSystem

load_dotenv()


def get_model_from_provider():
    provider = os.getenv("MODEL_PROVIDER", "openai").lower()
    model_name = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
    temperature = float(os.getenv("MODEL_TEMPERATURE", "0"))
    llm = None

    if provider == "openai":
        llm = ChatOpenAI(model=model_name, temperature=temperature)
    elif provider == "vertexai":
        project_id = os.getenv("PROJECT_ID")
        location = os.getenv("LOCATION", "us-west1")
        api_key = os.getenv("VERTEX_API_KEY")
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            project=project_id,
            location=location,
            api_key=api_key,
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")
    return llm


class AgentCore:
    def __init__(self):
        self.llm = get_model_from_provider()
        self.rag = RAGSystem()
        self.agent_executor = None

    def setup(self):
        """
        Sets up the Agent with a retrieval tool as described in the documentation.
        """
        retriever = self.rag.get_retriever()

        @tool
        def retrieve_knowledge(query: str) -> str:
            """
            Search the knowledge base for domain-specific information.
            Use this when you need facts about the company or technical details.
            """
            docs = retriever.invoke(query)
            return "\n\n".join([doc.page_content for doc in docs])

        tools = [retrieve_knowledge]

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful Slack assistant. Use the retrieve_knowledge tool to answer questions based on the knowledge base. If you don't know the answer, say you don't know.",
                ),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        agent = create_tool_calling_agent(self.llm, tools, prompt)

        self.agent_executor = AgentExecutor(
            agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
        )

    def ask(self, query: str) -> str:
        """
        Processes a query through the RAG agent.
        """
        if not self.agent_executor:
            self.setup()

        response = self.agent_executor.invoke({"input": query})
        output = response["output"]

        if isinstance(output, list):
            text_parts = []
            for part in output:
                if isinstance(part, dict) and "text" in part:
                    text_parts.append(part["text"])
                else:
                    text_parts.append(str(part))
            return "".join(text_parts)

        return str(output)
