import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from langchain.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma
from tenacity import retry, stop_after_attempt, wait_exponential

# Configuration
@dataclass
class Config:
    chroma_path: Path = Path("chroma")
    model_name: str = "llama3.2"
    temperature: float = 0.7
    max_tokens: int = 500
    num_results: int = 5
    similarity_threshold: float = 0.7

PROMPT_TEMPLATE = """
Instructions: Using only the provided context, answer the question. If the answer cannot be found in the context, say "I cannot answer this based on the provided context."

Context:
{context}

Question: {question}

Please provide a clear and concise answer, citing specific information from the context where relevant.
"""

class RAGQueryEngine:
    def __init__(self, config: Config, embedding_function):
        """Initialize the RAG query engine with configuration and embedding function."""
        self.config = config
        self.embedding_function = embedding_function
        self.logger = self._setup_logging()
        self.db = self._initialize_db()
        self.prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        self.model = self._initialize_model()

    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger("RAGQueryEngine")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def _initialize_db(self) -> Chroma:
        """Initialize and verify the Chroma database."""
        try:
            db = Chroma(
                persist_directory=str(self.config.chroma_path),
                embedding_function=self.embedding_function
            )
            # Verify DB has documents
            if db._collection.count() == 0:
                raise ValueError("Database is empty. Please add documents first.")
            return db
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {str(e)}")
            raise

    def _initialize_model(self) -> OllamaLLM:
        """Initialize the language model with retry mechanism."""
        try:
            return OllamaLLM(
                model=self.config.model_name,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize model: {str(e)}")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _get_relevant_documents(self, query: str) -> List[Tuple[Document, float]]:
        """Retrieve relevant documents with retry mechanism."""
        results = self.db.similarity_search_with_score(
            query, 
            k=self.config.num_results
        )
        
        # Filter results based on similarity threshold
        filtered_results = [
            (doc, score) for doc, score in results 
            if score >= self.config.similarity_threshold
        ]
        
        if not filtered_results:
            self.logger.warning("No documents met the similarity threshold")
            return results[:1]  # Return at least one result
            
        return filtered_results

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def query(self, query_text: str) -> dict:
        """
        Process a query and return response with metadata.
        """
        try:
            self.logger.info(f"Processing query: {query_text}")
            
            # Get relevant documents
            results = self._get_relevant_documents(query_text)
            
            # Prepare context and prompt
            context_text = "\n\n---\n\n".join([
                f"[Source: {doc.metadata.get('id', 'Unknown')}]\n{doc.page_content}"
                for doc, _score in results
            ])
            
            prompt = self.prompt_template.format(
                context=context_text,
                question=query_text
            )
            
            # Generate response
            response_text = self.model.invoke(prompt)
            
            # Prepare response metadata
            sources = [
                {
                    'id': doc.metadata.get('id', 'Unknown'),
                    'score': float(score),
                    'title': doc.metadata.get('title', 'Unknown')
                }
                for doc, score in results
            ]
            
            return {
                'response': response_text,
                'sources': sources,
                'metadata': {
                    'num_sources': len(sources),
                    'model': self.config.model_name,
                    'temperature': self.config.temperature
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}")
            raise

def main():
    """Main entry point for the CLI application."""
    parser = argparse.ArgumentParser(description="RAG Query Engine")
    parser.add_argument("query_text", type=str, help="The query text to process")
    parser.add_argument("--model", type=str, default="llama3.2", help="Model name to use")
    parser.add_argument("--temperature", type=float, default=0.7, help="Model temperature")
    parser.add_argument("--max-tokens", type=int, default=500, help="Maximum tokens in response")
    parser.add_argument("--num-results", type=int, default=5, help="Number of results to retrieve")
    parser.add_argument("--similarity-threshold", type=float, default=0.7, help="Minimum similarity score")
    
    args = parser.parse_args()
    
    # Create configuration
    config = Config(
        model_name=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        num_results=args.num_results,
        similarity_threshold=args.similarity_threshold
    )
    
    try:
        # Initialize engine
        from get_embedding_function import get_embedding_function
        engine = RAGQueryEngine(config, get_embedding_function())
        
        # Process query
        result = engine.query(args.query_text)
        
        # Format and print response
        print("\nResponse:")
        print("-" * 80)
        print(result['response'])
        print("\nSources:")
        print("-" * 80)
        for source in result['sources']:
            print(f"- {source['title']} (ID: {source['id']}, Similarity: {source['score']:.3f})")
            
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()