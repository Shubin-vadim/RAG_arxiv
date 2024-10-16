# RAG Arxiv Project

👋🏻 Welcome to the RAG for Science papers Project repository!
This project is dedicated to the implementation of the RAG system for scientific articles. Poetry is used as a package management manager to ensure a smooth workflow and dependency handling.

## Quick Links

1. [Project Overview](#project-overview)
2. [Folder Structure Explained](#folder-structure-explained)
3. [Dataset Management](#dataset-management)
4. [Usage](#usage)

---

## Project Overview
The main goal of this project is to create a RAG system for scientific articles.

## Folder Structure Explained

The project's folder structure is designed for clarity and modularity. Here's an overview of the key folders and their contents:

- **src**:Python scripts for implementing RAG and working with data.
- **docs**: Documentation for the project.
- **notebooks**: Jupyter notebooks for EDA and preprocessing data.
- **reports** : Objects obtained as a result of EDA, in *.jpg and *.html formats.
- **pyproject.toml**: Manages project dependencies using Poetry package manager.

## Dataset Management
You can download the dataset from the [Kaggle](https://www.kaggle.com/datasets/Cornell-University/arxiv/data)

## Usage

### Preparing the Datasets
Before running the RAG process, ensure your datasets are preprocessed and structured properly. You must specify paths to the preprocessed datasets in your configuration YAML file under the key `interim_data`.Dataset should already contain a column specified by `prepared_column` in the configuration file, which in this case is `prepared_text`.

### Run the RAG
To start RAG, execute the `run.py` script with a YAML configuration file. Place the configuration file, such as `params.yaml`. For instance:

```bash
poetry run python run.py --config_path params.yaml
```
