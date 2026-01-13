install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

lint:
	# Add the current directory to the path so pylint can find imports
	export PYTHONPATH=$$(pwd) && pylint --disable=R,C,W0621 app/*.py

test:
	python -m pytest -vv app/test_main.py

build:
	docker build -t roberta-service .

all: install lint test

run:
	docker run -p 8080:8080 roberta-onnx-service