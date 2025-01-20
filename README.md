# How "lost in the middle"? Evaluating LLMs on information retrieval with long input contexts

This is the github repo for the blog post https://ycmak.substack.com/p/how-lost-in-the-middle-evaluating-long-context-llms -- all findings and charts in the blog post were generated with this code.

These folders contain material for the blog post:

* `/raw_responses`: Raw json responses from the retrieval tests.
* `/results`: Computed results for each model run.
* `/accuracy_data`: Detailed stats about each run, with information about each needle.
* `/charts`: Retrieval performance charts for each model run.
* `/agg_charts`: Aggregate charts showing performance for different models or for the same model across different runs.

## Getting started

Ensure you have poetry installed: https://python-poetry.org/docs/basic-usage/

Install this via poetry: `poetry install`.

Create a `.env` file in the project root, using `.env.template` as a template. Add in the various API keys.

To run a test, run `python3 retrieval_test.py`.

To generate charts, run `python3 gen_charts.py`.

To regenerate data, run `python3 gen_data.py`.

## Todo

* Add CLI interface for retrieval_test.py and gen_charts.py.

