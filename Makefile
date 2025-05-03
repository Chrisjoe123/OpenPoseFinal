.PHONY: clean clean-test clean-pyc clean-build docs help
.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

try:
	from urllib import pathname2url
except:
	from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -fr {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

format:
	pipenv run black .

lint: ## check style with pylint
	pipenv run black --check .
#	pipenv run pytype tf_pose
#	pipenv run pipenv-setup check
#	pipenv run python -m pytest --pylint --pylint-error-types=EFCRW tf_pose

test: ## run tests quickly with the default Python
	pipenv run python -m pytest --dead-fixtures
	pipenv run python -m pytest -v -l tests

coverage: ## check code coverage quickly with the default Python
	coverage run --source tf_pose -m pytest
	coverage report -m
	coverage html
	$(BROWSER) htmlcov/index.html

bumpversion:
	git config user.name "bumpvserion"
	git config user.email "bumpversion@detalytics.com"
	pipenv run bumpversion minor  --allow-dirty
	git push origin master --tags

release: ## package and upload a release
	pipenv run  twine upload --repository pypi dist/*

config: clean ## install the package to the active Python's site-packages
	source .env; pipenv install --dev