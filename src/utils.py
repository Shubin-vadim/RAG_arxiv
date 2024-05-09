import json
import yaml
import pandas as pd
import os
import sys
from dotenv import load_dotenv

def transform_from_json_to_csv(data_path: str, save_path: str, columns: list[str]) -> None:

    """
    Transform data from a JSON file to a CSV file.

    Args:
        data_path (str): The path to the JSON data file.
        save_path (str): The path to save the CSV file.
        columns (list[str]): List of column names to include in the CSV.

    Returns:
        None
    """

    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line)
            current_data = []
            for column in columns:
                if column in doc:
                    current_data.append(doc[column])

    df_data = pd.DataFrame(current_data, columns=columns)
    df_data.to_csv(save_path, index=False)
    print(f'Data saved to {save_path}')

def load_config_json(config_path) -> dict:
    """
    Load configuration settings from a JSON file.

    Args:
        config_path (str): The path to the JSON configuration file.

    Returns:
        dict: Dictionary containing the configuration settings.

    Raises:
        FileNotFoundError: If the specified configuration file is not found.
        json.JSONDecodeError: If there is an error parsing the JSON file.
    """

    try:
        with open(config_path) as config_file:
            return json.load(config_file)
    except FileNotFoundError:
        sys.exit(f"Configuration file not found: {config_path}")
    except json.JSONDecodeError:
        sys.exit(f"Error parsing JSON file: {config_path}")

def load_config_yaml(config_path) -> dict:

    """
    Load configuration settings from a YAML file.

    Args:
        config_path (str): The path to the YAML configuration file.

    Returns:
        dict: Dictionary containing the configuration settings.

    Raises:
        FileNotFoundError: If the specified configuration file is not found.
        yaml.YAMLError: If there is an error parsing the YAML file.
    """

    try:
        with open(config_path) as config_file:
            return yaml.safe_load(config_file)
    except FileNotFoundError:
        sys.exit(f"Configuration file not found: {config_path}")
    except json.JSONDecodeError:
        sys.exit(f"Error parsing YAML file: {config_path}")


def load_environment_variables(project_path) -> str:
    """
    Load environment variables from a .env file located in the project directory.

    Args:
        project_path (str): The path to the project directory.

    Returns:
        str: LLAMA_API_TOKEN extracted from the .env file.

    Raises:
        FileNotFoundError: If the .env file is not found in the specified project directory.
        ValueError: If LLAMA_API_TOKEN is not found in the .env file.
    """

    dotenv_path = os.path.join(project_path, ".env")
    if not os.path.exists(dotenv_path):
        raise FileNotFoundError(f".env file not found in {dotenv_path}")
    load_dotenv(dotenv_path)
    token = os.getenv("LLAMA_API_TOKEN")
    if token is None:
        raise ValueError("LLAMA_API_TOKEN not found in.env file")
    return token
