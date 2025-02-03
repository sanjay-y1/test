import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

class PDFRAGBot:
    def __init__(self, pdf_folder: str, db_dir: str = "db"):
        self.pdf_folder = pdf_folder
        self.db_dir = db_dir
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.llm = OllamaLLM(
            model="llama3.2",
            callbacks=[StreamingStdOutCallbackHandler()],
            temperature=0.1
        )
        
    def load_pdfs(self) -> List:
        documents = []
        for file in os.listdir(self.pdf_folder):
            if file.endswith('.pdf'):
                pdf_path = os.path.join(self.pdf_folder, file)
                loader = PyPDFLoader(pdf_path)
                documents.extend(loader.load())
        return documents
    
    def split_documents(self, documents: List, chunk_size: int = 1000, chunk_overlap: int = 200) -> List:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        return text_splitter.split_documents(documents)
    
    def create_vectorstore(self, splits: List) -> Chroma:
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=self.db_dir
        )
        return vectorstore
    
    def setup_qa_chain(self, vectorstore: Chroma) -> RetrievalQA:
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 3}
        )
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
    
    def initialize(self):
        documents = self.load_pdfs()
        splits = self.split_documents(documents)
        vectorstore = self.create_vectorstore(splits)
        return self.setup_qa_chain(vectorstore)
    
    def load_existing_db(self):
        vectorstore = Chroma(
            persist_directory=self.db_dir,
            embedding_function=self.embeddings
        )
        return self.setup_qa_chain(vectorstore)

def main():
    # Initialize the bot
    pdf_folder = "data"  # folder containing your PDFs
    bot = PDFRAGBot(pdf_folder)
    
    # Create new vector database or load existing one
    if not os.path.exists("db"):
        qa_chain = bot.initialize()
    else:
        qa_chain = bot.load_existing_db()
    
    # Interactive QA loop
    while True:
        question = input("\nEnter your question (or 'quit' to exit): ")
        if question.lower() == 'quit':
            break
            
        result = qa_chain.invoke({"query": question})
        print("\nAnswer:", result["result"])
        print("\nSources:")
        for doc in result["source_documents"]:
            print(f"- Page {doc.metadata['page']}: {doc.metadata['source']}")

if __name__ == "__main__":
    main()