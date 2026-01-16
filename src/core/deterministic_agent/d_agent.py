import os
import asyncio
from dotenv import load_dotenv
import io
import contextlib
from langchain.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langchain_google_genai import ChatGoogleGenerativeAI
from src.rag.graphiti import GraphitiClient
import nest_asyncio
from src.core.deterministic_agent.utils import evaluate_query_results, off_topic_response
from src.core.deterministic_agent.utils import GeneratedQueries, SelectedIndexes, State, WorkerInput, QueryEvaluation
from src.core.deterministic_agent.utils import IndexMetadata

load_dotenv(override=True)


model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=1.0,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    vertexai=True
)

graphitiClient = GraphitiClient()

def filterer(state: State):
    """
    Agent that determines if the user question is related to the medical field.
    """
    prompt = [
        SystemMessage(content="""You are a medical domain classifier. 
        Determine if the user's question is related to health, medicine, biology, 
        diseases, or clinical research.
        
        Respond with 'YES' if it is medical-related, and 'NO' if it is off-topic."""),
        HumanMessage(content=f"User Question: {state['user_question']}")
    ]
    
    response = model.invoke(prompt)
    decision = response.content.strip().upper()
    
    is_medical = "YES" in decision
       
    return {"is_medical": is_medical}


def orchestrator(state: State):
    """
    Orchestrator that analyzes the user question and available indexes,
    then selects which indexes should be processed by workers.
    """
    
    index_selector = model.with_structured_output(SelectedIndexes)
    
    indexes_info = "\n\n".join([
        f"Index ID: {idx.index_id}\n"
        f"Title: {idx.title}\n"
        f"Year: {idx.year}\n"
        f"Summary: {idx.summary}"
        for idx in state["available_indexes"]
    ])
    
    selected = index_selector.invoke([
        SystemMessage(content="""You are an expert research assistant. 
        Analyze the user's question and select which indexes from the available 
        indexes would be most relevant to answer it. 
        
        Consider:
        - Topical relevance to the question
        - Time period relevance if the question has temporal aspects
        - Potential for containing useful information
        
        Select between 1-5 most relevant indexes."""),
        HumanMessage(content=f"""User Question: {state['user_question']}

Available Indexes:
{indexes_info}

Select the most relevant indexes to answer this question.""")
    ])
    
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
    index_id = state['index_metadata'].index_id
    
    try:
        await graphitiClient.initialize()
    except Exception as e:
        print(f"  ✗ Failed to initialize Graphiti client: {e}")
        return {"worker_outputs": []}
    
    query_gen_llm = model.with_structured_output(GeneratedQueries)
    
    generated = query_gen_llm.invoke([
        SystemMessage(content="""You are an expert at generating search queries.
        Given a user question and information about a document index, generate 3-5
        diverse search queries that will retrieve relevant information to answer 
        the question.
        
        Make queries specific and varied to capture different aspects."""),
        HumanMessage(content=f"""User Question: {state['user_question']}

Index Information:
Title: {state['index_metadata'].title}
Summary: {state['index_metadata'].summary}
Year: {state['index_metadata'].year}

Generate 3-5 search queries for this index.""")
    ])
    
    query_strings = [q.query for q in generated.queries]
    all_results = []
    
    for query in query_strings:
        try:
            edges = await graphitiClient.search(query)
            
            for i, edge in enumerate(edges[:5]):
                score = 0.1 + (i * 0.05)
                
                all_results.append({
                    "text": edge["fact"],
                    "metadata": {
                        "source": edge.get("source_title", "Graphiti Knowledge Graph"),
                        "topic": "General", 
                        "year": state['index_metadata'].year 
                    },
                    "score": score,
                    "query": query
                })
        except Exception as e:
            print(f"  ✗ Error searching for '{query}': {e}")

    results_by_query = {}
    for result in all_results:
        query = result['query']
        if query not in results_by_query:
            results_by_query[query] = []
        results_by_query[query].append(result)
    
    evaluated_queries = []
    relevant_chunks = []
    
    for query_obj in generated.queries:
        query_str = query_obj.query
        results = results_by_query.get(query_str, [])
        
        evaluation = evaluate_query_results(results, min_relevant=2, score_threshold=0.6)
        
        if evaluation.is_relevant:
            evaluated_queries.append(query_obj)
            relevant_chunks.extend([r for r in results if r['score'] < 0.6])

    if len(evaluated_queries) == 0:
        return {"worker_outputs": []}
    
    seen_texts = set()
    unique_chunks = []
    for chunk in relevant_chunks:
        text = chunk['text']
        if text not in seen_texts:
            seen_texts.add(text)
            unique_chunks.append(chunk)
    
    unique_chunks.sort(key=lambda x: x['score'])
    
    top_chunks = unique_chunks[:10]
    chunks_context = "\n\n---\n\n".join([
        f"Source: {chunk['metadata']['source']}\n"
        f"Topic: {chunk['metadata']['topic']}\n"
        f"Relevance Score: {chunk['score']:.3f}\n"
        f"Content: {chunk['text']}"
        for chunk in top_chunks
    ])
    
    summary_response = model.invoke([
        SystemMessage(content="""You are an expert research assistant. 
        Create a focused summary of the provided documents that directly answers 
        the user's question. 
        
        Structure your summary to:
        1. Highlight key developments/findings relevant to the question
        2. Maintain chronological awareness (note the year when relevant)
        3. Be specific and concrete
        4. Keep it concise but informative (2-3 paragraphs)
        
        Do NOT include a references section - that will be added separately."""),
        HumanMessage(content=f"""User Question: {state['user_question']}

        Retrieved Documents from {state['index_metadata'].title} ({state['index_metadata'].year}):

        {chunks_context}

        Create a summary that answers the user's question based on these documents.""")
            ])
    
    summary = summary_response.content

    sources_used = []
    seen_sources = set()
    
    for chunk in top_chunks:
        source = chunk['metadata']['source']
        if source not in seen_sources:
            seen_sources.add(source)
            sources_used.append({
                'source': source,
                'topic': chunk['metadata']['topic'],
                'year': chunk['metadata']['year'],
                'score': chunk['score']
            })
    

    worker_output = {
        "index_id": state["index_metadata"].index_id,
        "title": state["index_metadata"].title,
        "year": state["index_metadata"].year,
        "summary": summary,
        "queries_generated": len(generated.queries),
        "queries_passed": len(evaluated_queries),
        "chunks_retrieved": len(unique_chunks),
        "sources": sources_used 
    }
    
    return {"worker_outputs": [worker_output]}


def synthesizer(state: State):
    """
    Final node that takes all worker outputs and produces the final timeline 
    with consistent section formatting throughout.
    """
    
    sorted_outputs = sorted(state["worker_outputs"], key=lambda x: x["year"])
    
    if not sorted_outputs:
        print("\n✗ No workers returned results - cannot generate timeline")
        error_message = ("="*80 + "\n" +
                        "TIMELINE\n" +
                        "="*80 + "\n" +
                        "Unfortunately, no relevant information was found in the available indexes to answer this question. "
                        "This could mean:\n"
                        "- The question requires information not covered in the indexed documents\n"
                        "- The search queries did not match the available content well enough\n"
                        "- The relevance threshold filtered out all potential results\n\n"
                        "Please try:\n"
                        "- Rephrasing your question\n"
                        "- Asking about topics more directly covered in the indexes\n"
                        "- Checking if additional indexes need to be added\n" +
                        "="*80 + "\n" +
                        "RESEARCH LIMITATIONS\n" +
                        "="*80 + "\n" +
                        "This search was attempted across the available indexes, but no documents met the relevance threshold. "
                        "All generated queries returned either no results or results with insufficient relevance scores.\n" +
                        "="*80 + "\n" +
                        "SOURCES & REFERENCES\n" +
                        "="*80 + "\n" +
                        "No sources were retrieved for this query.\n")
        return {"final_timeline": error_message}
    
    worker_context = "\n\n".join([
        f"=== {output['title']} ({output['year']}) ===\n"
        f"Index: {output['index_id']}\n"
        f"Queries: {output['queries_passed']}/{output['queries_generated']} passed evaluation\n"
        f"Chunks: {output['chunks_retrieved']} retrieved\n"
        f"Sources: {len(output['sources'])} documents\n\n"
        f"Summary:\n{output['summary']}"
        for output in sorted_outputs
    ])
    
    final_response = model.invoke([
        SystemMessage(content="""You are an expert research synthesizer.
        Create a concise timeline that answers the user's question by 
        synthesizing information from multiple sources.
        
        Your timeline should:
        1. Present information chronologically using bullet points
        2. Focus only on the most critical developments
        3. Be brief and to the point
        4. usage of bolding for key terms
        
        Do NOT include limitations or references sections."""),
        HumanMessage(content=f"""User Question: {state['user_question']}

        Research gathered from {len(sorted_outputs)} sources:

        {worker_context}

        Create a concise timeline that synthesizes this information.""")
            ])
    
    timeline_content = final_response.content
    
    indexes_used = [output['title'] for output in sorted_outputs]
    years_covered = [output['year'] for output in sorted_outputs]
    total_chunks = sum(o['chunks_retrieved'] for o in sorted_outputs)
    passed_queries = sum(o['queries_passed'] for o in sorted_outputs)
    
    limitations_response = model.invoke([
        SystemMessage(content="""You are an expert research analyst.
        Create a very brief limitations statement (1-2 sentences) about the research scope.
        Mention what years/indexes were covered.
        Do NOT include any title or heading."""),
        HumanMessage(content=f"""User Question: {state['user_question']}
        
        - Indexes: {', '.join(indexes_used)}
        - Years: {min(years_covered)} to {max(years_covered)}
        
        Generate a brief limitations statement.""")
            ])
    
    limitations_content = limitations_response.content

    # Simplified References
    all_sources = set()
    for output in sorted_outputs:
        for source_info in output['sources']:
            all_sources.add(source_info['source'])
    
    references_content = "*Sources:*"
    for source in sorted(all_sources):
        references_content += f"\n• {source}"
    
    # Construct concise final output
    final_output = f"*TIMELINE*\n{timeline_content}\n\n"
    final_output += f"*LIMITATIONS*\n{limitations_content}\n\n"
    final_output += f"{references_content}"
    
    return {"final_timeline": final_output}


def assign_workers(state: State):
    """
    Conditional edge function that creates a worker for each selected index.
    Uses the Send API to dispatch work in parallel.
    """
    index_map = {idx.index_id: idx for idx in state["available_indexes"]}
    
    return [
        Send("worker_node", {
            "user_question": state["user_question"],
            "index_metadata": index_map[selected.index_id],
            "selected_index": selected,
        })
        for selected in state["selected_indexes"]]


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
        {
            "orchestrator": "orchestrator",
            "off_topic": "off_topic"
        }
    )
    graph_builder.add_edge("off_topic", END)
    graph_builder.add_conditional_edges(
        "orchestrator",
        assign_workers,
        ["worker_node"]
    )
    graph_builder.add_edge("worker_node", "synthesizer")
    graph_builder.add_edge("synthesizer", END)
    
    return graph_builder.compile()


class DeterministicAgent:
    def __init__(self):
        self.graph = build_research_timeline_graph()
        self._loop = None
    
    def setup(self):
        """Setup async capabilities if needed"""
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None

    def ask(self, query: str, thread_id: str = None) -> str:
        """
        Synchronous wrapper for ask_async
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.ask_async(query, thread_id))
        else:
            try:
                nest_asyncio.apply()
            except ImportError:
                pass
            return loop.run_until_complete(self.ask_async(query, thread_id))

    async def ask_async(self, query: str, thread_id: str = None) -> str:
        """
        Process a query using the deterministic agent graph.
        Returns the final timeline as a string.
        """

        default_indexes = [
            IndexMetadata(
                index_id="graphiti_main",
                title="Medical Knowledge Graph",
                year="2024",
                summary="A comprehensive medical knowledge graph containing information about diseases, treatments, and clinical research."
            )
        ]

        initial_state = {
            "user_question": query,
            "available_indexes": default_indexes,
            "selected_indexes": [],
            "is_medical": False,
            "worker_outputs": [],
            "final_timeline": ""
        }
 
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            result = await self.graph.ainvoke(initial_state)
        
        logs = f.getvalue()
        
        return {
            "answer": result.get("final_timeline", "No response generated."),
            "logs": logs
        }