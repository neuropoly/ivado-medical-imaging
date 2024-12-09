# Testing

We have divided our tests into two categories, `functional_tests` and `unit_tests`. In each
folder, you will find a `t_utils` file with some helper functions, and a `t_template` file,
which provides a testing template.

## Running in GitHub

Checkout `ivadomed/.github/workflows/run_tests.yml` to see how tests are run on pull requests.

## Running Locally

1. Install dependencies
```
cd ivadomed  # root of the repo
pip install -e .[dev]
```

2. To run all tests:
```
pytest -v
```

or, to run specific tests:
```
pytest -v testing/functional_tests/
pytest -v testing/unit_tests/
pytest -v testing/functional_tests/test_example.py
```

## Wiki

You can read more about our testing here: https://github.com/ivadomed/ivadomed/wiki/Tests
