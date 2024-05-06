import logging
import sys
import pandas as pd
import argparse
from src.DataLoader import DataLoader
from src.VectorStoreService import VectorStoreService
from src.LlamaServiceAPI import LlamaServiceAPI
from src.config import LLAMA_MODEL, LLAMA_API_TOKEN, EMBEDDING_MODEL, RERANK_MODEL
from src.Rerankers import RerankerColBERT
from src.prompts_template import user_template, refine_template
from src.utils import load_config_yaml

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

def main(config_path: str) -> None:

    config = load_config_yaml(config_path=config_path)

    df = pd.read_csv(config['data']['interim_data'])

    debug = config['rag']['debug']
    count_records = config['rag']['count_records']

    if debug:
        df = df.head(count_records)

    data_loader = DataLoader(df=df, prepared_column=config['data']['prepared_column'])

    documents = data_loader.get_documents()

    llm = LlamaServiceAPI(
        model=LLAMA_MODEL,
        api_key=LLAMA_API_TOKEN,
        temperature=config['llm']['temperature'],
        max_tokens=config['llm']['max_tokens']
    )

    reranker = RerankerColBERT()

    vectore_service = VectorStoreService(
        documents=None,
        show_progress=True,
        llm=llm.get_llm(),
        embedding_model=EMBEDDING_MODEL,
        chroma_path='./DB',
        name_collection='arxiv',
        cache_folder='./models',
        node_postprocessors=reranker.get_reranker(),
        similarity_top_k=config['vector_store']['similarity_top_k'],
        alpha=config['vector_store']['alpha'],
    )

    vectore_service.update_prompts(
        prompt_template=user_template,
        refine_template=refine_template,
    )

    while True:
         query = input('Enter your question')
         answer = vectore_service.send_query(query)
         print(answer)
         print('-----' * 10)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='RAG using a YAML configuration file.'
    )

    parser.add_argument(
        '--config_path', type=str, required=True, help='Path to the YAML config file.'
    )

    args = parser.parse_args()

    main(args.config_path)
