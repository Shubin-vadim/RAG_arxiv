from llama_index.core import Document
import pandas as pd

class DataLoader():

    """
    Class for loading data into memory.

    Attributes:
        df (pd.DataFrame): DataFrame containing the data.
        prepared_column (str): Name of the column containing prepared text. Default is 'prepared_text'.
        arxiv_documents (list): List to store ArXiv documents.

    Methods:
        get_documents(): Get the list of documents.
        get_count_documents(): Get the count of documents.
    """

    def __init__(self, df: pd.DataFrame = None,
                  prepared_column: str = 'prepared_text'
                ) -> None:

        """
        Initialize a DataLoader instance.

        Args:
            df (pd.DataFrame): DataFrame containing the data. Default is None.
            prepared_column (str): Name of the column containing prepared text. Default is 'prepared_text'.

        Returns:
            None
        """

        self.df = df
        self.prepared_column = prepared_column
        self.arxiv_documents = []

    def get_documents(self) -> list:
        """
        Get the list of documents.

        Returns:
            list: List of ArXiv documents.
        """

        self.arxiv_documents = [Document(text=item) for item in list(self.df[self.prepared_column])]
        return self.arxiv_documents

    def get_count_documents(self) -> int:
        """
        Get the count of documents.

        Returns:
            int: The count of ArXiv documents.
        """

        return len(self.arxiv_documents)
