.PHONY: style quality

FILES := src/config.py src/DataLoader.py src/LlamaServiceAPI.py src/preprocessing.py src/prompts_template.py src/Rerankers.py src/utils.py src/VectorStoreService.py

style:
	for file in $(FILES); do \
    	python3 -m black --line-length 119 $$file; \
    	python3 -m isort $$file; \
    	ruff check --fix $$file; \
		mypy  $$file; \
  	done

quality:
	for file in $(FILES); do \
    	python3 -m black --check --line-length 119 $$file; \
    	python3 -m isort --check-only $$file; \
    	ruff check $$file; \
	done
