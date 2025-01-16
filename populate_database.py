import argparse
import os
import shutil
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any
import hashlib
from datetime import datetime

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_chroma import Chroma
from get_embedding_function import get_embedding_function

@dataclass
class ProcessingConfig:
    chunk_size: int = 1000
    chunk_overlap: int = 200
    min_chunk_size: int = 100
    separator_chars: List[str] = None
    clean_text: bool = True
    extract_metadata: bool = True
    
    def __post_init__(self):
        if self.separator_chars is None:
            self.separator_chars = ["\n\n", "\n", ".", "!", "?", ";"]

class DocumentProcessor:
    def __init__(
        self,
        data_path: Path,
        chroma_path: Path,
        config: ProcessingConfig,
        logger: Optional[logging.Logger] = None
    ):
        self.data_path = Path(data_path)
        self.chroma_path = Path(chroma_path)
        self.config = config
        self.logger = logger or self._setup_logging()
        self.embedding_function = get_embedding_function()

    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger("DocumentProcessor")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def clean_text_content(self, text: str) -> str:
        """Clean and normalize text content."""
        import re
        
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters while preserving essential punctuation
        text = re.sub(r'[^\w\s.,!?;:()-]', '', text)
        
        # Normalize quotes
        text = re.sub(r'[''′`]', "'", text)
        text = re.sub(r'[""″]', '"', text)
        
        # Remove unnecessary line breaks
        text = re.sub(r'\n+', '\n', text)
        
        return text.strip()

    def extract_basic_metadata(self, document: Document) -> Dict[str, Any]:
        """Extract basic metadata without requiring additional dependencies."""
        metadata = document.metadata.copy()
        
        # Extract filename and extension
        source_path = Path(metadata.get('source', ''))
        metadata.update({
            'filename': source_path.name,
            'file_extension': source_path.suffix.lower(),
            'processed_date': datetime.now().isoformat(),
        })
        
        # Calculate content hash for versioning
        content_hash = hashlib.md5(document.page_content.encode()).hexdigest()
        metadata['content_hash'] = content_hash
        
        return metadata

    def load_documents(self) -> List[Document]:
        """Load documents with enhanced error handling and validation."""
        self.logger.info(f"Loading documents from {self.data_path}")
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_path}")
        
        document_loader = PyPDFDirectoryLoader(
            str(self.data_path),
            extract_images=False  # Disable image extraction to avoid dependency issues
        )
        
        try:
            documents = document_loader.load()
            self.logger.info(f"Loaded {len(documents)} documents")
            return documents
        except Exception as e:
            self.logger.error(f"Error loading documents: {e}")
            raise

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents with improved chunking strategy."""
        self.logger.info("Splitting documents into chunks")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            length_function=len,
            separators=self.config.separator_chars,
            is_separator_regex=False,
        )
        
        chunks = text_splitter.split_documents(documents)
        
        # Post-process chunks
        processed_chunks = []
        for chunk in chunks:
            # Skip chunks that are too small
            if len(chunk.page_content) < self.config.min_chunk_size:
                continue
                
            # Clean text if configured
            if self.config.clean_text:
                chunk.page_content = self.clean_text_content(chunk.page_content)
                
            # Add basic metadata
            chunk.metadata = self.extract_basic_metadata(chunk)
                
            processed_chunks.append(chunk)
        
        self.logger.info(f"Created {len(processed_chunks)} chunks")
        return processed_chunks

    def calculate_chunk_ids(self, chunks: List[Document]) -> List[Document]:
        """Calculate unique chunk IDs with improved metadata."""
        last_page_id = None
        current_chunk_index = 0
        
        for chunk in chunks:
            source = chunk.metadata.get("source", "unknown")
            page = chunk.metadata.get("page", 0)
            content_hash = chunk.metadata.get("content_hash", "")
            
            current_page_id = f"{source}:{page}"
            
            if current_page_id == last_page_id:
                current_chunk_index += 1
            else:
                current_chunk_index = 0
                
            chunk_id = f"{current_page_id}:{current_chunk_index}:{content_hash[:8]}"
            last_page_id = current_page_id
            
            chunk.metadata["id"] = chunk_id
            chunk.metadata["chunk_index"] = current_chunk_index
            
        return chunks

    def add_to_chroma(self, chunks: List[Document]):
        """Add documents to Chroma with deduplication and versioning."""
        self.logger.info("Adding documents to Chroma database")
        
        db = Chroma(
            persist_directory=str(self.chroma_path),
            embedding_function=self.embedding_function
        )
        
        # Get existing documents
        existing_items = db.get(include=["metadatas"])
        existing_ids = set(existing_items["ids"])
        existing_hashes = {
            meta.get("content_hash"): id
            for id, meta in zip(existing_items["ids"], existing_items["metadatas"])
            if meta and "content_hash" in meta
        }
        
        self.logger.info(f"Found {len(existing_ids)} existing documents")
        
        # Prepare new chunks
        new_chunks = []
        updated_chunks = []
        
        for chunk in chunks:
            chunk_id = chunk.metadata["id"]
            content_hash = chunk.metadata.get("content_hash")
            
            if chunk_id in existing_ids:
                continue
                
            if content_hash in existing_hashes:
                updated_chunks.append(chunk)
                db.delete(ids=[existing_hashes[content_hash]])
            else:
                new_chunks.append(chunk)
        
        if new_chunks:
            self.logger.info(f"Adding {len(new_chunks)} new documents")
            new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
            db.add_documents(new_chunks, ids=new_chunk_ids)
        
        if updated_chunks:
            self.logger.info(f"Updating {len(updated_chunks)} existing documents")
            updated_chunk_ids = [chunk.metadata["id"] for chunk in updated_chunks]
            db.add_documents(updated_chunks, ids=updated_chunk_ids)
        
        if not new_chunks and not updated_chunks:
            self.logger.info("No new or updated documents to add")

    def clear_database(self):
        """Clear the database with safety checks."""
        if self.chroma_path.exists():
            self.logger.info(f"Clearing database at {self.chroma_path}")
            try:
                shutil.rmtree(self.chroma_path)
                self.logger.info("Database cleared successfully")
            except Exception as e:
                self.logger.error(f"Error clearing database: {e}")
                raise

def main():
    parser = argparse.ArgumentParser(description="Document Processing and Embedding Pipeline")
    parser.add_argument("--reset", action="store_true", help="Reset the database")
    parser.add_argument("--data-path", type=str, default="data", help="Path to data directory")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size for splitting")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Chunk overlap size")
    parser.add_argument("--no-clean", action="store_true", help="Disable text cleaning")
    
    args = parser.parse_args()
    
    config = ProcessingConfig(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        clean_text=not args.no_clean
    )
    
    processor = DocumentProcessor(
        data_path="data",
        chroma_path="chroma",
        config=config
    )
    
    try:
        if args.reset:
            processor.clear_database()
        
        documents = processor.load_documents()
        chunks = processor.split_documents(documents)
        chunks_with_ids = processor.calculate_chunk_ids(chunks)
        processor.add_to_chroma(chunks_with_ids)
        
    except Exception as e:
        logging.error(f"Error processing documents: {e}")
        raise

if __name__ == "__main__":
    main()