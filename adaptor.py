import environ
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
    UnstructuredCSVLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQAWithSourcesChain 
from langchain_community.vectorstores import FAISS
import langchain_huggingface
from pathlib import Path



env = environ.Env()
environ.Env.read_env()


class Adaptor:
    def __init__(self):
        self.llm_text = env("LLM_TYPE")
        if self.llm_text.lower() == "openai":
            from langchain_openai import OpenAIEmbeddings, OpenAI
            from langchain_openai import ChatOpenAI
            self.llm = OpenAI(temperature=0, openai_api_key=env("OPENAI_API_KEY"))
            self.prompt = ChatPromptTemplate.from_template(
                "answer the following request: {topic}"
            )
            self.llm_chat = ChatOpenAI(
                temperature=0.4, openai_api_key=env("OPENAI_API_KEY"),model_name=env("OPENAI_MODEL", default="gpt-4o-mini")
            )
            self.embedding = OpenAIEmbeddings(model="text-embedding-ada-002")
        elif self.llm_text.lower() == "local":
            self.ollama_url = env("OLLAMA_URL")
            self.local_model = env("LOCAL_MODEL")
            from langchain_ollama import ChatOllama
            from langchain_huggingface import HuggingFaceEmbeddings
            self.prompt = ChatPromptTemplate.from_template(
                "answer the following request: {topic}"
            )
            self.llm_chat = ChatOllama(
                base_url=self.ollama_url, model=self.local_model
            )
            model_name = "BAAI/bge-small-en"
            model_kwargs = {"device": "cpu"}
            encode_kwargs = {"normalize_embeddings": True}
            
            self.embedding = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
            )
        elif self.llm_text.lower() == "hybrid":
            from langchain_huggingface import HuggingFaceBgeEmbeddings
            self.llm = OpenAI(temperature=0.3, openai_api_key=env("OPENAI_API_KEY"))
            model_name = "BAAI/bge-small-en"
            model_kwargs = {"device": "cpu"}
            encode_kwargs = {"normalize_embeddings": True}
            self.embedding = HuggingFaceBgeEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
            )
        else:
            raise ValueError("Invalid LLM")

    def load_document(self, filename):
        loaders = {
            ".pdf": PyPDFLoader,
            ".txt": TextLoader,
            ".csv": UnstructuredCSVLoader,
            ".doc": UnstructuredWordDocumentLoader,
            ".docx": UnstructuredWordDocumentLoader,
            ".md": UnstructuredMarkdownLoader,
            ".odt": UnstructuredODTLoader,
            ".ppt": UnstructuredPowerPointLoader,
            ".pptx": UnstructuredPowerPointLoader,
            ".xlsx": UnstructuredExcelLoader,
        }
        for extension, loader_cls in loaders.items():
            if filename.endswith(extension):
                loader = loader_cls(filename)
                documents = loader.load()
                break
        else:
            raise ValueError("Invalid file type")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=30
        )
        return text_splitter.split_documents(documents=documents)
    
    def vector_doc(self, filename):
        doc = self.load_document(filename)
        retriever = FAISS.from_documents(doc, self.embedding).as_retriever()
        return retriever
    
    def query_doc(self, query, filename):
        print(f"file: {filename}")
        retriever = self.vector_doc(filename)
        # if self.llm_text.lower() == "openai":
        qa = RetrievalQAWithSourcesChain.from_chain_type(
                llm=self.llm_chat, chain_type="stuff", retriever=retriever, verbose=True
            )
        result = qa.invoke(query)
        answer = result["answer"].replace("\n", "")
        source = result["sources"]
        return f"Answer: {answer}\nSource: {source}"
    
    def chat(self, query):
        print(f"adaptor query: {query}")
        from langchain_core.output_parsers import StrOutputParser
        chain = self.prompt | self.llm_chat | StrOutputParser()
        result = chain.invoke({"topic": query})
        return result
    
    def query_datastore(self, query):
        print("Entered function")
        try:
            # Check if the FAISS index files exist in the vector_store directory
            if not (Path("vector_store/index.faiss").exists() and Path("vector_store/index.pkl").exists()):
                return "No documents have been added to the datastore yet."

            # Load the FAISS vector store from the specified directory
            retriever = FAISS.load_local("vector_store", self.embedding, allow_dangerous_deserialization=True).as_retriever()

            print("Retriever loaded successfully")

            # Create the QA chain with the loaded retriever
            qa = RetrievalQAWithSourcesChain.from_chain_type(
                llm=self.llm_chat, chain_type="stuff", retriever=retriever, verbose=True
            )

            # Query the retriever using the input query
            result = qa.invoke(query)  # Directly passing the query to the QA chain
            result = result['answer']  # Extract the answer from the result dictionary
            return result

        except Exception as e:
            print(f"An error occurred: {e}")
            return f"An error occurred: {e}"
    
    def add_to_datastore(self, filename):
        try:
            doc = self.load_document(f"{filename}")
            vectorstore_path = Path("vector_store")
            
            # Check if the vector_store directory exists and contains the index files
            if not (vectorstore_path.exists() and (vectorstore_path / "index.faiss").exists() and (vectorstore_path / "index.pkl").exists()):
                print("Vector store does not exist, creating a new one.")
                vectorstore = FAISS.from_documents(doc, self.embedding)
                vectorstore.save_local("vector_store")  # Save directly to the directory
            else:
                print("Vector store exists, loading and merging.")
                existing_vectorstore = FAISS.load_local("vector_store", self.embedding, allow_dangerous_deserialization=True)
                new_vectorstore = FAISS.from_documents(doc, self.embedding)
                existing_vectorstore.merge_from(new_vectorstore)
                existing_vectorstore.save_local("vector_store")  # Save the merged index back to the same directory

            print(f"Successfully added {filename} to the datastore.")
        except Exception as e:
            print(f"An error occured: {e}")
            return(f"An error occurred: {e}")