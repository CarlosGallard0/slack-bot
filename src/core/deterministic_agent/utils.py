from typing import Annotated, List, TypedDict
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
import operator

class IndexMetadata(BaseModel):
    """Metadata for a single index in the RAPTOR RAG system"""
    index_id: str = Field(description="Unique identifier for the index")
    title: str = Field(description="Title of the index")
    summary: str = Field(description="Summary of the content in this index")
    year: int = Field(description="Year associated with this index content")


class SelectedIndex(BaseModel):
    """An index selected by the orchestrator for processing"""
    index_id: str = Field(description="ID of the selected index")
    relevance_reasoning: str = Field(description="Why this index is relevant to the query")


class SelectedIndexes(BaseModel):
    """List of indexes selected for processing"""
    indexes: List[SelectedIndex] = Field(
        description="Indexes that are relevant to answering the user's question"
    )


class GeneratedQuery(BaseModel):
    """A single query to send to the database"""
    query: str = Field(description="The search query to send to the vector database")
    reasoning: str = Field(description="Why this query will help answer the question")


class GeneratedQueries(BaseModel):
    """List of queries generated for a specific index"""
    queries: List[GeneratedQuery] = Field(
        description="3-5 queries to retrieve relevant information",
        min_items=3,
        max_items=5
    )


class QueryEvaluation(BaseModel):
    """Evaluation of a query's relevance"""
    is_relevant: bool = Field(description="Whether the query returned relevant chunks")
    relevant_chunk_count: int = Field(description="Number of relevant chunks retrieved")

class State(TypedDict):
    """Main graph state"""
    user_question: str  # The original user question
    available_indexes: List[IndexMetadata]  # All available indexes
    selected_indexes: List[SelectedIndex]  # Indexes selected by orchestrator
    worker_outputs: Annotated[list, operator.add]  # All workers write here in parallel
    final_timeline: str  # Final synthesized answer
    is_medical: bool


class WorkerInput(TypedDict):
    """Input passed to each worker via Send API"""
    user_question: str
    index_metadata: IndexMetadata
    selected_index: SelectedIndex