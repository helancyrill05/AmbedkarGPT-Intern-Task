"""
Professional RAG System - Production Ready
Multi-document support, error handling, caching, metrics, and more
"""

import os
import sys
import json
import logging
import warnings
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import hashlib

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
os.environ['ANONYMIZED_TELEMETRY'] = 'False'
os.environ['PYTHONWARNINGS'] = 'ignore'

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Configure logging with UTF-8 encoding for Windows
import io

# Set logging level to ERROR for noisy libraries
logging.getLogger('chromadb').setLevel(logging.ERROR)
logging.getLogger('langchain').setLevel(logging.ERROR)
logging.getLogger('sentence_transformers').setLevel(logging.ERROR)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_system.log', encoding='utf-8'),
        logging.StreamHandler(io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8'))
    ]
)
logger = logging.getLogger(__name__)

# Disable ChromaDB telemetry
os.environ['ANONYMIZED_TELEMETRY'] = 'False'


class Config:
    """Configuration management"""
    
    # Paths
    DATA_DIR = Path("data")
    VECTOR_DB_DIR = Path("vector_db")
    CACHE_FILE = Path("document_cache.json")
    FALLBACK_FILE = Path("speech.txt")  # Check root directory too
    
    # Model settings
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_MODEL = "mistral"
    LLM_TEMPERATURE = 0.2  # Lower for factual responses
    
    # Chunking settings
    CHUNK_SIZE = 500  # Smaller chunks for better retrieval
    CHUNK_OVERLAP = 100
    
    # Retrieval settings
    TOP_K = 3
    SIMILARITY_THRESHOLD = 0.3  # Lower threshold for small documents
    
    # Supported file types
    SUPPORTED_EXTENSIONS = {'.txt', '.md'}
    
    @classmethod
    def ensure_directories(cls):
        """Create necessary directories"""
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.VECTOR_DB_DIR.mkdir(exist_ok=True)


class DocumentLoader:
    """Enhanced document loading with multiple format support"""
    
    @staticmethod
    def load_documents(data_dir: Path) -> List[Document]:
        """Load all supported documents from directory"""
        documents = []
        
        # Try data directory first
        if data_dir.exists():
            files = list(data_dir.glob("*"))
            supported_files = [f for f in files if f.suffix in Config.SUPPORTED_EXTENSIONS]
        else:
            supported_files = []
        
        # Fallback: check for speech.txt in root directory
        if not supported_files and Config.FALLBACK_FILE.exists():
            logger.warning(f"⚠️  No data/ folder found. Using {Config.FALLBACK_FILE} from root directory")
            logger.warning(f"💡 Tip: Create a 'data/' folder and put documents there for better organization")
            supported_files = [Config.FALLBACK_FILE]
        
        if not supported_files:
            raise ValueError(f"No supported files found. Please:\n"
                           f"  1. Create a 'data/' folder\n"
                           f"  2. Add .txt or .md files to it\n"
                           f"  OR place 'speech.txt' in the root directory")
        
        logger.info(f"Found {len(supported_files)} documents to load")
        
        for filepath in supported_files:
            try:
                doc = DocumentLoader._load_single_file(filepath)
                documents.extend(doc)
                logger.info(f"[OK] Loaded: {filepath.name}")
            except Exception as e:
                logger.error(f"✗ Failed to load {filepath.name}: {e}")
        
        return documents
    
    @staticmethod
    def _load_single_file(filepath: Path) -> List[Document]:
        """Load a single file with proper encoding handling"""
        encodings = ['utf-8', 'latin-1', 'cp1252']
        
        for encoding in encodings:
            try:
                with open(filepath, 'r', encoding=encoding) as f:
                    content = f.read()
                
                if not content.strip():
                    logger.warning(f"Empty file: {filepath.name}")
                    return []
                
                return [Document(
                    page_content=content,
                    metadata={
                        "source": filepath.name,
                        "filepath": str(filepath),
                        "loaded_at": datetime.now().isoformat(),
                        "size": len(content)
                    }
                )]
            except UnicodeDecodeError:
                continue
        
        raise ValueError(f"Could not decode {filepath.name} with any supported encoding")


class DocumentCache:
    """Cache system to avoid re-processing unchanged documents"""
    
    @staticmethod
    def get_file_hash(filepath: Path) -> str:
        """Generate hash of file content"""
        with open(filepath, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    @staticmethod
    def load_cache() -> Dict:
        """Load cache from file"""
        if not Config.CACHE_FILE.exists():
            return {}
        
        try:
            with open(Config.CACHE_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load cache: {e}")
            return {}
    
    @staticmethod
    def save_cache(cache: Dict):
        """Save cache to file"""
        try:
            with open(Config.CACHE_FILE, 'w') as f:
                json.dump(cache, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save cache: {e}")
    
    @staticmethod
    def check_if_changed(data_dir: Path) -> bool:
        """Check if any documents have changed"""
        cache = DocumentCache.load_cache()
        current_hashes = {}
        
        for filepath in data_dir.glob("*"):
            if filepath.suffix in Config.SUPPORTED_EXTENSIONS:
                current_hashes[str(filepath)] = DocumentCache.get_file_hash(filepath)
        
        if cache.get('file_hashes') != current_hashes:
            cache['file_hashes'] = current_hashes
            DocumentCache.save_cache(cache)
            return True
        
        return False


class EnhancedTextSplitter:
    """Improved text splitting with better semantic preservation"""
    
    @staticmethod
    def split_documents(documents: List[Document]) -> List[Document]:
        """Split documents into chunks with overlap"""
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
            add_start_index=True
        )
        
        chunks = splitter.split_documents(documents)
        
        # Add chunk metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "chunk_id": i,
                "chunk_size": len(chunk.page_content)
            })
        
        logger.info(f"[OK] Split into {len(chunks)} chunks")
        return chunks


class VectorStoreManager:
    """Manages vector store with caching and updates"""
    
    def __init__(self):
        self.embeddings = None
        self.vectorstore = None
    
    def setup(self, documents: List[Document], force_rebuild: bool = False) -> Chroma:
        """Setup or load vector store"""
        
        # Check if we need to rebuild
        should_rebuild = force_rebuild or DocumentCache.check_if_changed(Config.DATA_DIR)
        
        logger.info("Loading embedding model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        if Config.VECTOR_DB_DIR.exists() and not should_rebuild:
            logger.info("Loading existing vector store...")
            try:
                self.vectorstore = Chroma(
                    persist_directory=str(Config.VECTOR_DB_DIR),
                    embedding_function=self.embeddings
                )
                logger.info("[OK] Vector store loaded from cache")
                return self.vectorstore
            except Exception as e:
                logger.warning(f"Could not load cached store: {e}, rebuilding...")
        
        logger.info("Building new vector store...")
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=str(Config.VECTOR_DB_DIR)
        )
        logger.info("[OK] Vector store created")
        
        return self.vectorstore


class RAGSystem:
    """Main RAG system orchestrator"""
    
    def __init__(self):
        self.qa_chain = None
        self.vectorstore = None
        self.query_history = []
        
        # Custom prompt template
        self.prompt_template = """Use the following context to answer the question. 
If you cannot find the answer in the context, say "I don't have enough information to answer this question based on the provided documents."

IMPORTANT: Only state facts that are explicitly mentioned in the context. Do not infer or guess information that is not directly stated, especially regarding authorship, names, or attributions.

Be specific and cite which document the information comes from when possible.

Context:
{context}

Question: {question}

Detailed Answer:"""
    
    def setup(self, force_rebuild: bool = False):
        """Initialize the complete RAG pipeline"""
        
        try:
            Config.ensure_directories()
            
            # Load documents
            logger.info("=" * 80)
            logger.info("Loading documents...")
            documents = DocumentLoader.load_documents(Config.DATA_DIR)
            
            if not documents:
                raise ValueError("No documents loaded")
            
            logger.info(f"[OK] Loaded {len(documents)} document(s)")
            
            # Split documents
            logger.info("\nSplitting documents...")
            chunks = EnhancedTextSplitter.split_documents(documents)
            
            # Setup vector store
            logger.info("\nSetting up vector store...")
            vector_manager = VectorStoreManager()
            self.vectorstore = vector_manager.setup(chunks, force_rebuild)
            
            # Initialize LLM
            logger.info("\nInitializing LLM...")
            llm = Ollama(
                model=Config.LLM_MODEL,
                temperature=Config.LLM_TEMPERATURE,
                callbacks=[StreamingStdOutCallbackHandler()]
            )
            logger.info("[OK] LLM initialized")
            
            # Create custom prompt
            PROMPT = PromptTemplate(
                template=self.prompt_template,
                input_variables=["context", "question"]
            )
            
            # Create QA chain with simpler retrieval for small documents
            logger.info("\nCreating retrieval chain...")
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=self.vectorstore.as_retriever(
                    search_type="similarity",  # Changed from similarity_score_threshold
                    search_kwargs={"k": Config.TOP_K}
                ),
                return_source_documents=True,
                chain_type_kwargs={"prompt": PROMPT}
            )
            logger.info("[OK] RAG pipeline ready!")
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"Failed to setup RAG system: {e}")
            raise
    
    def ask(self, question: str) -> Dict:
        """Ask a question and get detailed response with sources"""
        
        if not self.qa_chain:
            raise RuntimeError("RAG system not initialized. Call setup() first.")
        
        if not question.strip():
            return {"error": "Empty question"}
        
        try:
            logger.info(f"\n{'=' * 80}")
            logger.info(f"Question: {question}")
            logger.info("-" * 80)
            
            # Get response
            result = self.qa_chain.invoke({"query": question})
            
            # Extract sources
            sources = []
            if result.get('source_documents'):
                for doc in result['source_documents']:
                    sources.append({
                        "source": doc.metadata.get('source', 'Unknown'),
                        "content": doc.page_content[:200] + "...",
                        "chunk_id": doc.metadata.get('chunk_id', 'N/A')
                    })
            
            response = {
                "question": question,
                "answer": result['result'],
                "sources": sources,
                "timestamp": datetime.now().isoformat()
            }
            
            # Log to history
            self.query_history.append(response)
            
            # Display sources
            if sources:
                logger.info("\nSources:")
                for i, source in enumerate(sources, 1):
                    logger.info(f"  [{i}] {source['source']} (chunk {source['chunk_id']})")
            else:
                logger.warning("No source documents found - LLM may be hallucinating!")
            
            logger.info("=" * 80)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            return {"error": str(e)}
    
    def save_history(self, filename: str = "query_history.json"):
        """Save query history to file"""
        try:
            with open(filename, 'w') as f:
                json.dump(self.query_history, f, indent=2)
            logger.info(f"[OK] History saved to {filename}")
        except Exception as e:
            logger.error(f"Could not save history: {e}")


def interactive_mode(rag: RAGSystem):
    """Run interactive Q&A session"""
    
    print("\n" + "=" * 80)
    print("Interactive Mode")
    print("=" * 80)
    print("Commands:")
    print("  'quit' or 'exit' - Exit the program")
    print("  'history' - Show query history")
    print("  'save' - Save query history to file")
    print("  'stats' - Show system statistics")
    print("=" * 80 + "\n")
    
    while True:
        try:
            question = input("\nYour question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\nThank you for using the RAG System!")
                break
            
            if question.lower() == 'history':
                print("\nQuery History:")
                for i, q in enumerate(rag.query_history, 1):
                    print(f"  {i}. {q['question']}")
                continue
            
            if question.lower() == 'save':
                rag.save_history()
                continue
            
            if question.lower() == 'stats':
                print(f"\nStatistics:")
                print(f"  Total queries: {len(rag.query_history)}")
                print(f"  Vector store size: {rag.vectorstore._collection.count()} chunks")
                continue
            
            if not question:
                print("Please enter a question.")
                continue
            
            rag.ask(question)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error: {e}")


def main():
    """Main execution"""
    
    print("=" * 80)
    print("Professional RAG System - Production Ready")
    print("=" * 80 + "\n")
    
    try:
        # Initialize RAG system
        rag = RAGSystem()
        rag.setup(force_rebuild=False)  # Set to True to force rebuild
        
        # Example questions
        example_questions = [
            "What is the main argument of the text?",
            "What solutions are proposed?",
            "Summarize the key points in 3 sentences."
        ]
        
        print("\n" + "=" * 80)
        print("Testing with Example Questions")
        print("=" * 80)
        
        for question in example_questions:
            rag.ask(question)
        
        # Interactive mode
        interactive_mode(rag)
        
        # Save history on exit
        rag.save_history()
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()