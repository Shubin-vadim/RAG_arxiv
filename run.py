import logging
import sys
import pandas as pd
from src.DataLoader import DataLoader
from src.VectorStoreService import VectorStoreService
from src.LlamaServiceAPI import LlamaServiceAPI
from src.config import LLAMA_MODEL, LLAMA_API_TOKEN, EMBEDDING_MODEL, RERANK_MODEL
from src.Rerankers import RerankerColBERT
from src.prompts_template import user_template, refine_template
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

def main(
         data_path: str,
         prepared_column: str,
         debug=False,
         count_records=30,
         ) -> None:

    df = pd.read_csv(data_path)

    if debug:
        df = df.head(count_records)

    data_loader = DataLoader(df=df, prepared_column=prepared_column)
    documents = data_loader.get_documents()

    llm = LlamaServiceAPI(
        model=LLAMA_MODEL,
        api_key=LLAMA_API_TOKEN,
    )

    reranker = RerankerColBERT()
    vectore_service = VectorStoreService(
        documents=documents,
        show_progress=True,
        llm=llm.get_llm(),
        embedding_model=EMBEDDING_MODEL,
        chroma_path='./DB',
        name_collection='axriv',
        cache_folder='./models',
        **{
            'similarity_top_k': 5,
            'alpha': 0.5,
            'node_postprocessors': reranker.get_reranker
        }
    )

    vectore_service.update_prompts(
        prompt_template=user_template,
        refine_template=refine_template
    )

    while True:
         query = input('Enter your question')
         answer = vectore_service.send_query(query)
         print(answer)

if __name__ == '__main__':
    main(
        data_path='data\interim\prepared_data.csv',
        prepared_column='prepared_text',
        debug=True,
        )
