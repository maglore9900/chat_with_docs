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
from langchain_community.vectorstores import Qdrant
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQAWithSourcesChain 
from qdrant_client.models import VectorParams, Distance
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore

env = environ.Env()
environ.Env.read_env()


class Adaptor:
    def __init__(self):
        self.llm_text = env("LLM_TYPE")
        self.ollama_url = env("OLLAMA_URL")
        self.local_model = env("LOCAL_MODEL")
        self.qIP = env("QDRANT_IP").split(":")[0]
        self.qP = env("QDRANT_IP").split(":")[1]
        if self.llm_text.lower() == "openai":
            from langchain_openai import OpenAIEmbeddings, OpenAI
            from langchain_openai import ChatOpenAI
            self.llm = OpenAI(temperature=0, openai_api_key=env("OPENAI_API_KEY"))
            self.prompt = ChatPromptTemplate.from_template(
                "answer the following request: {topic}"
            )
            self.llm_chat = ChatOpenAI(
                temperature=0.3, openai_api_key=env("OPENAI_API_KEY")
            )
            self.embedding = OpenAIEmbeddings(model="text-embedding-ada-002")
        elif self.llm_text.lower() == "local":
            from langchain_community.llms import Ollama
            from langchain_community.embeddings import HuggingFaceBgeEmbeddings
            from langchain_community.chat_models import ChatOllama
            self.llm = Ollama(base_url=self.ollama_url, model=self.local_model)
            self.prompt = ChatPromptTemplate.from_template(
                "answer the following request: {topic}"
            )
            self.llm_chat = ChatOllama(
                base_url=self.ollama_url, model=self.local_model
            )
            model_name = "BAAI/bge-small-en"
            model_kwargs = {"device": "cpu"}
            encode_kwargs = {"normalize_embeddings": True}
            self.embedding = HuggingFaceBgeEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
            )
        elif self.llm_text.lower() == "hybrid":
            from langchain_community.embeddings import HuggingFaceBgeEmbeddings
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
        qdrant = Qdrant.from_documents(
            doc,
            self.embedding,
            location=":memory:"
        )
        retriever = qdrant.as_retriever()
        return retriever
    
    def query_doc(self, query, retriever):
        # if self.llm_text.lower() == "openai":
        qa = RetrievalQAWithSourcesChain.from_chain_type(
                llm=self.llm, chain_type="stuff", retriever=retriever, verbose=True
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
    
    def add_to_datastore(self, filename, vector_db):
        try:
            # Convert document into an embedding
            embedding = self.emctor_doc(filename)

            # Connect to Qdrant (adjust host/port for a persistent instance)
            client = QdrantClient(host=self.qIP, port=self.qP)  # Use appropriate connection settings

            # Check if collection exists
            try:
                collection_info = client.get_collection(vector_db)
                print(f"Collection '{vector_db}' already exists. Adding to existing collection.")
            except Exception:
                # If collection doesn't exist, create a new one
                print(f"Collection '{vector_db}' does not exist. Creating a new collection.")
                client.recreate_collection(
                    collection_name=vector_db,  # Use vector_db as the collection name
                    vectors_config=VectorParams(size=3072, distance=Distance.COSINE)
                )

            # Save the embedding to the vector store (either in new or existing collection)
            vector_store = QdrantVectorStore(
                client=client,
                collection_name=vector_db,
                embeddings=[embedding],  # A list of embeddings
                payload=[{'filename': filename}]  # Metadata associated with the vector
            )
            print(f"Successfully added {filename} to the datastore.")
        except Exception as e:
            print(f"An error occurred: {e}")
    
    def query_datastore(self, query, datastore):
        try:
            # Connect to Qdrant (adjust host/port for a persistent instance)
            client = QdrantClient(host=self.qIP, port=self.qP)  # Use appropriate connection settings

            # Perform the search in the vector collection
            search_results = client.search(
                collection_name=datastore,
                query_vector=query,
                limit=5  
            )
            qa = RetrievalQAWithSourcesChain.from_chain_type(
                llm=self.llm, chain_type="stuff", retriever=search_results, verbose=True
            )
            result = qa.invoke(query)
            return result
        except Exception as e:
            print(e)