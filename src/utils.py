import json
import pandas as pd
import os
import sys
import argparse
from dotenv import load_dotenv

def transform_from_json_to_csv(data_path: str, save_path: str, columns: list[str]) -> None:
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

def load_config(config_path):
    try:
        with open(config_path) as config_file:
            return json.load(config_file)
    except FileNotFoundError:
        sys.exit(f"Configuration file not found: {config_path}")
    except json.JSONDecodeError:
        sys.exit(f"Error parsing JSON file: {config_path}")


def load_environment_variables(project_path):
    dotenv_path = os.path.join(project_path, ".env")
    if not os.path.exists(dotenv_path):
        raise FileNotFoundError(f".env file not found in {dotenv_path}")
    load_dotenv(dotenv_path)
    token = os.getenv("LLAMA_API_TOKEN")
    if token is None:
        raise ValueError("LLAMA_API_TOKEN not found in.env file")
    return token


def load_configuration():
    """Load configuration from command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run model inference, calculate the accuracy."
    )
    parser.add_argument(
        "--repo_id", type=str ,required=False, help="Path to the model from huggingface."
    )
    parser.add_argument(
        "--config_path", type=str, required=False, help="Path to the config JSON file"
    )
    parser.add_argument(
        "--questions_path", type=str, required=False, help="Path to the questions JSON file"
    )
    parser.add_argument(
        "--results", type=str, required=True, help="Path to save the results from model in CSV file."
    )
    parser.add_argument(
        "--parser_results", type=str, required=True, help="Path to save the results in JSON file."
    )
    parser.add_argument(
        "--test_data", type=str, required=True, help="Path to the test dataframe CSV file."
    )
    parser.add_argument(
        "--num_rows",type=int, help="Number of rows to process from the test dataframe.",
    )
    parser.add_argument(
        "--size_chunk",type=int, help="Chunk size.",
    )
    return parser.parse_args()
