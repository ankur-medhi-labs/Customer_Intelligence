.PHONY: install ingest batch-analyze themes eval test

install:
	pip install -e .
	pip install -r requirements.txt

ingest:
	ci ingest

batch-analyze:
	ci batch-analyze

themes:
	ci themes

eval:
	ci eval

test:
	pytest -q
