import os
import json
import asyncio
from dotenv import load_dotenv
from langchain.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langchain_google_genai import ChatGoogleGenerativeAI
from src.rag.adapter import RAGAdapter
import nest_asyncio
from src.core.deterministic_agent.utils import (
    evaluate_query_results,
    off_topic_response,
)
from src.core.deterministic_agent.utils import (
    GeneratedQueries,
    SelectedIndexes,
    State,
    WorkerInput,
)
from src.core.deterministic_agent.utils import IndexMetadata


def clean_source_name(source: str) -> str:
    """
    Clean chunked source names to extract the base file name.
    Removes chunk numbers and timestamps from paths like:
    new_docs/41377627_Title.pdf_1_1768420545.461907 -> new_docs/41377627_Title.pdf
    """
    if ".pdf" not in source:
        return source

    # Split by '_' and find the part with .pdf
    parts = source.split("_")
    for i, part in enumerate(parts):
        if ".pdf" in part:
            # Include up to and including the .pdf part
            base = "_".join(parts[: i + 1])
            return base
    return source


load_dotenv(override=True)


model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=1.0,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    vertexai=True,
)

rag_adapter = RAGAdapter(os.getenv("RAG_TYPE", "graphiti"))


def filterer(state: State):
    """
    Agent that determines if the user question is related to the medical field
    and decides whether to use Graphiti (timeline) or Raptor (facts).
    """
    prompt = [
        SystemMessage(
            content="""You are a medical domain classifier and RAG router. 
        
        1. Determine if the user's question is related to health, medicine, biology, 
           diseases, or clinical research.
        
        2. Decide the retrieval strategy:
           - Choose 'graphiti' if the question asks for a timeline, progression of disease, historical development, or sequence of events.
           - Choose 'raptor' if the question asks for specific facts, definitions, clinical data, or detailed information about a single point in time.
        
        Respond ONLY with a JSON object: {"is_medical": true/false, "strategy": "graphiti" or "raptor"}"""
        ),
        HumanMessage(content=f"User Question: {state['user_question']}"),
    ]

    response = model.invoke(prompt)
    try:
        # Simple cleanup if LLM returns markdown
        content = str(response.content).strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        decision = json.loads(content)
        is_medical = decision.get("is_medical", False)
        strategy = decision.get("strategy", "raptor")
    except:
        is_medical = "YES" in str(response.content).upper()
        strategy = "graphiti" if "timeline" in state["user_question"].lower() else "raptor"

    return {"is_medical": is_medical, "rag_type_decision": strategy}


def orchestrator(state: State):
    """
    Orchestrator that analyzes the user question and available indexes,
    then selects which indexes should be processed by workers.
    """

    index_selector = model.with_structured_output(SelectedIndexes)

    indexes_info = "\n\n".join(
        [
            f"Index ID: {idx.index_id}\n"
            f"Title: {idx.title}\n"
            f"Year: {idx.year}\n"
            f"Summary: {idx.summary}"
            for idx in state["available_indexes"]
        ]
    )

    selected = index_selector.invoke(
        [
            SystemMessage(
                content="""You are an expert research assistant. 
        Analyze the user's question and select which indexes from the available 
        indexes would be most relevant to answer it. 
        
        Consider:
        - Topical relevance to the question
        - Time period relevance if the question has temporal aspects
        - Potential for containing useful information
        
        Select between 1-5 most relevant indexes."""
            ),
            HumanMessage(
                content=f"""User Question: {state['user_question']}

Available Indexes:
{indexes_info}

Select the most relevant indexes to answer this question."""
            ),
        ]
    )

    return {"selected_indexes": selected.indexes}


async def worker_node(state: WorkerInput):
    """
    Complete worker pipeline that processes a single index:
    1. Initialize Graphiti client
    2. Generate 3-5 queries
    3. Execute searches against Graphiti
    4. Evaluate query quality
    5. Summarize relevant documents

    Returns worker output with summary and source references.
    """
    index_id = state["index_metadata"].index_id
    rag_strategy = state["rag_type_decision"]

    try:
        await rag_adapter.initialize(index_id)
    except Exception as e:
        print(f"  ✗ Failed to initialize RAG adapter for {index_id}: {e}")
        return {"worker_outputs": []}

    query_gen_llm = model.with_structured_output(GeneratedQueries)

    generated = query_gen_llm.invoke(
        [
            SystemMessage(
                content="""You are an expert at generating search queries.
        Given a user question and information about a document index, generate 3-5
        diverse search queries that will retrieve relevant information to answer 
        the question.
        
        Make queries specific and varied to capture different aspects."""
            ),
            HumanMessage(
                content=f"""User Question: {state['user_question']}

Index Information:
Title: {state['index_metadata'].title}
Summary: {state['index_metadata'].summary}
Year: {state['index_metadata'].year}

Generate 3-5 search queries for this index."""
            ),
        ]
    )

    query_strings = [q.query for q in generated.queries]
    all_results = []

    for query in query_strings:
        try:
            # We override the adapter's rag_type temporally for this search if needed
            original_rag = rag_adapter.rag_type
            rag_adapter.rag_type = rag_strategy
            results = await rag_adapter.search(query, index_id)
            rag_adapter.rag_type = original_rag

            for i, res in enumerate(results[:5]):
                # If the score is the default or coming from the adapter, we can use it
                # Graphiti implementation in adapter provides a mock score of 0.5
                # Raptor implementation provides real scores
                score = res.get("score", 0.1 + (i * 0.05))

                all_results.append(
                    {
                        "text": res["text"],
                        "metadata": {
                            "source": res["source"],
                            "topic": "General",
                            "year": state["index_metadata"].year,
                        },
                        "score": score,
                        "query": query,
                    }
                )
        except Exception as e:
            print(f"  ✗ Error searching for '{query}': {e}")

    results_by_query = {}
    for result in all_results:
        query = result["query"]
        if query not in results_by_query:
            results_by_query[query] = []
        results_by_query[query].append(result)

    evaluated_queries = []
    relevant_chunks = []

    for query_obj in generated.queries:
        query_str = query_obj.query
        results = results_by_query.get(query_str, [])

        evaluation = evaluate_query_results(
            results, min_relevant=2, score_threshold=0.6
        )

        if evaluation.is_relevant:
            evaluated_queries.append(query_obj)
            relevant_chunks.extend([r for r in results if r["score"] < 0.6])

    if len(evaluated_queries) == 0:
        return {"worker_outputs": []}

    seen_texts = set()
    unique_chunks = []
    for chunk in relevant_chunks:
        text = chunk["text"]
        if text not in seen_texts:
            seen_texts.add(text)
            unique_chunks.append(chunk)

    unique_chunks.sort(key=lambda x: x["score"])

    top_chunks = unique_chunks[:10]
    chunks_context = "\n\n---\n\n".join(
        [
            f"Source: {chunk['metadata']['source']}\n"
            f"Topic: {chunk['metadata']['topic']}\n"
            f"Relevance Score: {chunk['score']:.3f}\n"
            f"Content: {chunk['text']}"
            for chunk in top_chunks
        ]
    )

    summary_response = model.invoke(
        [
            SystemMessage(
                content="""You are an expert research assistant. 
        Create a focused summary of the provided documents that directly answers 
        the user's question. 
        
        Structure your summary to:
        1. Highlight key developments/findings relevant to the question
        2. Maintain chronological awareness (note the year when relevant)
        3. Be specific and concrete
        4. Keep it concise but informative (2-3 paragraphs)
        
        Do NOT include a references section - that will be added separately."""
            ),
            HumanMessage(
                content=f"""User Question: {state['user_question']}

        Retrieved Documents from {state['index_metadata'].title} ({state['index_metadata'].year}):

        {chunks_context}

        Create a summary that answers the user's question based on these documents."""
            ),
        ]
    )

    summary = str(summary_response.content)

    sources_used = []
    seen_sources = set()

    for chunk in top_chunks:
        source = chunk["metadata"]["source"]
        if source not in seen_sources:
            seen_sources.add(source)
            sources_used.append(
                {
                    "source": source,
                    "topic": chunk["metadata"]["topic"],
                    "year": chunk["metadata"]["year"],
                    "score": chunk["score"],
                }
            )

    worker_output = {
        "index_id": state["index_metadata"].index_id,
        "title": state["index_metadata"].title,
        "year": state["index_metadata"].year,
        "summary": summary,
        "queries_generated": len(generated.queries),
        "queries_passed": len(evaluated_queries),
        "chunks_retrieved": len(unique_chunks),
        "sources": sources_used,
    }

    return {"worker_outputs": [worker_output]}


def synthesizer(state: State):
    sorted_outputs = sorted(state["worker_outputs"], key=lambda x: x["year"])
    strategy = state["rag_type_decision"]

    if not sorted_outputs:
        print(f"\n✗ No workers returned results for {strategy}")
        return {"final_timeline": "No relevant information found.", "synthesized_answer": "I couldn't find specific facts to answer your question."}

    worker_context = "\n\n".join(
        [
            f"=== {output['title']} ({output['year']}) ===\n"
            f"Summary:\n{output['summary']}"
            for output in sorted_outputs
        ]
    )

    if strategy == "graphiti":
        # Timeline logic (existing)
        final_response = model.invoke(
            [
                SystemMessage(
                    content="""You are an expert research synthesizer focused on TIMELINES.
        Produce a chronological progression of events using bullet points.
        Return JSON: {"timeline": "...", "limitations": "..."}"""
                ),
                HumanMessage(
                    content=f"User Question: {state['user_question']}\n\nContext:\n{worker_context}"
                ),
            ]
        )
        # ... logic to parse timeline ...
    else:
        # Fact logic (Raptor)
        final_response = model.invoke(
            [
                SystemMessage(
                    content="""You are an expert medical researcher. 
        Provide a detailed, fact-based answer to the user's question based ONLY on the provided context.
        Return JSON: {"answer": "...", "limitations": "..."}"""
                ),
                HumanMessage(
                    content=f"User Question: {state['user_question']}\n\nContext:\n{worker_context}"
                ),
            ]
        )

    # Simplified parsing for both
    try:
        raw_content = str(final_response.content).strip()
        if "```" in raw_content:
            raw_content = raw_content.split("```")[1]
            if raw_content.startswith("json"): raw_content = raw_content[4:]
        parsed = json.loads(raw_content)
    except:
        parsed = {"timeline": str(final_response.content), "answer": str(final_response.content), "limitations": "Could not parse structured output."}

    all_sources = []
    for output in sorted_outputs:
        for s in output["sources"]:
            name = s["source"].split(".pdf")[0] + ".pdf"
            if name not in all_sources: all_sources.append(name)

    return {
        "final_timeline": parsed.get("timeline", ""),
        "synthesized_answer": parsed.get("answer", ""),
        "limitations": parsed.get("limitations", ""),
        "sources": sorted(all_sources),
    }


def assign_workers(state: State):
    """
    Conditional edge function that creates a worker for each selected index.
    Uses the Send API to dispatch work in parallel.
    """
    index_map = {idx.index_id: idx for idx in state["available_indexes"]}

    return [
        Send(
            "worker_node",
            {
                "user_question": state["user_question"],
                "index_metadata": index_map[selected.index_id],
                "selected_index": selected,
                "rag_type": state["rag_type"],
                "rag_type_decision": state["rag_type_decision"],
            },
        )
        for selected in state["selected_indexes"]
    ]


def route_after_filter(state: State):
    if state["is_medical"]:
        return "orchestrator"
    return "off_topic"


def build_research_timeline_graph():
    """Build and compile the complete research timeline graph"""

    graph_builder = StateGraph(State)

    graph_builder.add_node("filterer", filterer)
    graph_builder.add_node("orchestrator", orchestrator)
    graph_builder.add_node("off_topic", off_topic_response)
    graph_builder.add_node("worker_node", worker_node)
    graph_builder.add_node("synthesizer", synthesizer)

    graph_builder.add_edge(START, "filterer")
    graph_builder.add_conditional_edges(
        "filterer",
        route_after_filter,
        {"orchestrator": "orchestrator", "off_topic": "off_topic"},
    )
    graph_builder.add_edge("off_topic", END)
    graph_builder.add_conditional_edges("orchestrator", assign_workers, ["worker_node"])
    graph_builder.add_edge("worker_node", "synthesizer")
    graph_builder.add_edge("synthesizer", END)

    return graph_builder.compile()


class DeterministicAgent:
    def __init__(self, rag_type: str = None):
        self.rag_type = rag_type or os.getenv("RAG_TYPE", "graphiti")
        global rag_adapter
        rag_adapter = RAGAdapter(self.rag_type)
        self.graph = build_research_timeline_graph()

    def ask(self, query: str, thread_id: str = None, rag_type: str = None) -> dict:
        """
        Synchronous wrapper for ask_async
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.ask_async(query, thread_id, rag_type))
        else:
            try:
                nest_asyncio.apply()
            except ImportError:
                pass
            return loop.run_until_complete(self.ask_async(query, thread_id, rag_type))

    async def ask_async(self, query: str, thread_id: str = None, rag_type: str = None) -> dict:
        """
        Process a query using the deterministic agent graph.
        Returns the final timeline as a string.
        """

        # Default available indexes (we provide both sets so orchestrator can choose)
        default_indexes = [
            IndexMetadata(
                index_id="graphiti_main",
                title="Medical Knowledge Graph (Graphiti)",
                year=2024,
                summary="Knowledge graph for timelines and sequences of medical events.",
            ),
            IndexMetadata(
                index_id="raptor_main",
                title="Medical Fact Index (Raptor)",
                year=2024,
                summary="Hierarchical document index for specific medical facts and data.",
            )
        ]

        initial_state = {
            "user_question": query,
            "available_indexes": default_indexes,
            "selected_indexes": [],
            "is_medical": False,
            "worker_outputs": [],
            "final_timeline": "",
            "synthesized_answer": "",
            "rag_type": self.rag_type,
            "rag_type_decision": "raptor", # Default, will be updated by filterer
        }

        config = {"configurable": {"thread_id": thread_id}}
        result = await self.graph.ainvoke(initial_state, config=config)

        timeline = result.get("final_timeline", "").strip()
        answer = result.get("synthesized_answer", "").strip()

        return {
            "timeline": timeline,
            "answer": answer,
            "strategy_used": result.get("rag_type_decision"),
            "limitations": result.get("limitations", ""),
            "sources": result.get("sources", []),
        }
