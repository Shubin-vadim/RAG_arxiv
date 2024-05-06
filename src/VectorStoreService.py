from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import chromadb
import torch
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext

class VectorStoreService:

    def __init__(self,
                 documents = None,
                 show_progress=False,
                 llm=None,
                 embedding_model: str = '',
                 chroma_path: str = '.',
                 name_collection: str = '',
                 cache_folder: str = '.',
                 node_postprocessors=None,
                 similarity_top_k=3,
                 alpha=0.5,
                #  **kwargs: Dict[str, Any],
                 ) -> None:
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
        self.query_engine.update_prompts(
            {"response_synthesizer:text_qa_template": prompt_template, "response_synthesizer:refine_template": refine_template}
        )

    def send_query(self, message: str) -> str:
        return self.query_engine.query(message)
