from llama_index.core import Document
import pandas as pd

class DataLoader():
    def __init__(self, df: pd.DataFrame = None,
                  prepared_column: str = 'prepared_text'
                ) -> None:
        self.df = df
        self.prepared_column = prepared_column
        self.arxiv_documents = []

    def get_documents(self) -> list:
        self.arxiv_documents = [Document(text=item) for item in list(self.df[self.prepared_column])]
        return self.arxiv_documents

    def get_count_documents(self) -> int:
        return len(self.arxiv_documents)
