from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import chromadb
import torch
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext

class VectorStoreService:
    """
    Class to manage vector storage and query operations for retrieval-augmented generation (RAG).

    Args:
        documents (Optional[pd.DataFrame]): DataFrame containing documents to index. Default is None.
        show_progress (bool): Flag to indicate whether to show progress bars during indexing. Default is False.
        llm (Optional[LlamaServiceAPI]): Instance of LlamaServiceAPI for generating prompts. Default is None.
        embedding_model (str): Name of the embedding model to use. Default is an empty string.
        chroma_path (str): Path to the ChromaDB directory. Default is '.'.
        name_collection (str): Name of the collection in ChromaDB. Default is an empty string.
        cache_folder (str): Path to the cache folder for storing embeddings. Default is '.'.
        node_postprocessors (Optional[Reranker]): Instance of the reranker for post-processing nodes. Default is None.
        similarity_top_k (int): Number of top similar documents to retrieve. Default is 3.
        alpha (float): Alpha value for balancing original and reranked scores. Default is 0.5.

    Attributes:
        device (torch.device): Device used for embedding model.
        embed_model (HuggingFaceEmbedding): Instance of HuggingFaceEmbedding for embedding documents.
        chroma_client (PersistentClient): Client for accessing ChromaDB.
        chroma_collection (ChromaCollection): Collection in ChromaDB.
        vector_store (ChromaVectorStore): Vector store for storing document embeddings.
        storage_context (StorageContext): Context for storage operations.
        index (VectorStoreIndex): Index for vector storage.
        query_engine (RAGQueryEngine): Engine for executing RAG queries.

    Methods:
        update_prompts(prompt_template, refine_template): Update prompt templates for RAG query engine.
        send_query(message): Send a query to the RAG query engine and retrieve the response.
    """

    def __init__(self,
                 documents = None,
                 show_progress=False,
                 llm=None,
                 embedding_model: str = '',
                 chroma_path: str = '.',
                 name_collection: str = '',
                 cache_folder: str = '.',
                 node_postprocessors = None,
                 similarity_top_k: int = 3,
                 alpha: float = 0.5,
                 ) -> None:

        """
        Initialize a VectorStoreService instance.

        Args:
            documents (Optional[pd.DataFrame]): DataFrame containing documents to index. Default is None.
            show_progress (bool): Flag to indicate whether to show progress bars during indexing. Default is False.
            llm (Optional[LlamaServiceAPI]): Instance of LlamaServiceAPI for generating prompts. Default is None.
            embedding_model (str): Name of the embedding model to use. Default is an empty string.
            chroma_path (str): Path to the ChromaDB directory. Default is '.'.
            name_collection (str): Name of the collection in ChromaDB. Default is an empty string.
            cache_folder (str): Path to the cache folder for storing embeddings. Default is '.'.
            node_postprocessors (Optional[Reranker]): Instance of the reranker for post-processing nodes. Default is None.
            similarity_top_k (int): Number of top similar documents to retrieve. Default is 3.
            alpha (float): Alpha value for balancing original and reranked scores. Default is 0.5.

        Returns:
            None
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embed_model = HuggingFaceEmbedding(
            model_name=embedding_model,
            cache_folder=cache_folder,
            device= self.device
            )
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.chroma_collection = self.chroma_client.get_or_create_collection(name_collection)
        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        if documents is None:
            self.index = VectorStoreIndex.from_vector_store(
                self.vector_store,
                storage_context=self.storage_context,
                embed_model=self.embed_model,
                show_progress=show_progress,
                )
        else:
            self.index = VectorStoreIndex.from_documents(
                documents=documents,
                storage_context=self.storage_context,
                embed_model=self.embed_model,
                show_progress=show_progress,
                )
        self.query_engine = self.index.as_query_engine(
            llm=llm,
             node_postprocessors=[node_postprocessors],
            similarity_top_k=similarity_top_k,
            alpha=alpha,
        )

    def update_prompts(self, prompt_template, refine_template) -> None:
        """
        Update prompt templates for the RAG query engine.

        Args:
            prompt_template (str): Template for generating prompts.
            refine_template (str): Template for refining generated responses.

        Returns:
            None
        """

        self.query_engine.update_prompts(
            {"response_synthesizer:text_qa_template": prompt_template, "response_synthesizer:refine_template": refine_template}
        )

    def send_query(self, message: str) -> str:
        """
        Send a query message to the RAG query engine and retrieve the response.

        Args:
            message (str): The query message.

        Returns:
            str: The response generated by the RAG query engine.
        """

        return self.query_engine.query(message)
