from llama_index.postprocessor.colbert_rerank import ColbertRerank
from llama_index.core.postprocessor import SentenceTransformerRerank

class RerankerColBERT:

    """
    Reranker using ColBERT.

    Attributes:
        colbert_reranker (ColbertRerank): Instance of ColbertRerank for reranking sentences.

    Methods:
        get_reranker(): Get the reranker instance.
    """

    def __init__(self,
                 device='cpu',
                 top_n=3,
                 model='colbert-ir/colbertv2.0',
                 tokenizer='colbert-ir/colbertv2.0',
                 keep_retrieval_score=True,
                 ) -> None:
        """
        Initialize a RerankerColBERT instance.

        Args:
            device (str): Device to use for inference. Default is 'cpu'.
            top_n (int): Number of top candidates to rerank. Default is 3.
            model (str): Pre-trained model to use for reranking. Default is 'colbert-ir/colbertv2.0'.
            tokenizer (str): Pre-trained tokenizer to use for reranking. Default is 'colbert-ir/colbertv2.0'.
            keep_retrieval_score (bool): Flag to indicate whether to keep the retrieval score. Default is True.

        Returns:
            None
        """

        self.colbert_reranker = ColbertRerank(
            model=model,
            tokenizer=tokenizer,
            top_n=top_n,
            keep_retrieval_score=keep_retrieval_score,
            device=device
        )

    def get_reranker(self) -> ColbertRerank:
        """
        Get the reranker instance.

        Returns:
            ColbertRerank: The reranker instance.
        """
        return self.colbert_reranker

class RerankerSentenceTransformer:

    """
    Reranker using SentenceTransformer.

    Attributes:
        sentence_reranker (SentenceTransformerRerank): Instance of SentenceTransformerRerank for reranking sentences.

    Methods:
        get_reranker(): Get the reranker instance.
    """

    def __init__(self,
                 device='cpu',
                 top_n=3,
                 model='mixedbread-ai/mxbai-rerank-xsmall-v1',
                 keep_retrieval_score=True,
                 ) -> None:
        """
        Initialize a reranker instance.

        Args:
            device (str): Device to use for inference. Default is 'cpu'.
            top_n (int): Number of top candidates to rerank. Default is 3.
            model (str): Pre-trained model to use for reranking.
            keep_retrieval_score (bool): Flag to indicate whether to keep the retrieval score.

        Returns:
            None
        """

        self.sentence_reranker = SentenceTransformerRerank(
            model=model,
            top_n=top_n,
            keep_retrieval_score=keep_retrieval_score,
            device=device
        )

    def get_reranker(self) -> SentenceTransformerRerank:

        """
        Get the reranker instance.

        Returns:
            SentenceTransformerRerank: The reranker instance.
        """
        return self.sentence_reranker
