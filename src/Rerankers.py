from llama_index.postprocessor.colbert_rerank import ColbertRerank

class RerankerColBERT:
    def __init__(self,
                 device='cpu',
                 top_n=3,
                 model='colbert-ir/colbertv2.0',
                 tokenizer='colbert-ir/colbertv2.0',
                 keep_retrieval_score=True,
                 ) -> None:

        self.colbert_reranker = ColbertRerank(
            model=model,
            tokenizer=tokenizer,
            top_n=top_n,
            keep_retrieval_score=keep_retrieval_score,
            device=device
        )

    def get_reranker(self):
        return self.colbert_reranker

class RerankerSentenceTransformer:
    def __init__(self,
                 device='cpu',
                 top_n=3,
                 model='colbert-ir/colbertv2.0',
                 tokenizer='colbert-ir/colbertv2.0',
                 keep_retrieval_score=True,
                 ) -> None:

        self.sentence_reranker = ColbertRerank(
            model=model,
            tokenizer=tokenizer,
            top_n=top_n,
            keep_retrieval_score=keep_retrieval_score,
            device=device
        )
    def get_reranker(self):
        return self.sentence_reranker
