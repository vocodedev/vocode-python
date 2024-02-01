# Contributing to Vocode

Hi there! Thank you for wanting to contribute to Vocode! We're an open source project and are extremely open to contributions like new features,integrations, or documentation.

To contribute, please ["fork and pull request"](https://docs.github.com/en/get-started/quickstart/contributing-to-projects).

This project uses [Poetry](https://python-poetry.org/) as a dependency manager. Check out Poetry's [documentation on how to install it](https://python-poetry.org/docs/#installation) on your system before proceeding.

To install requirements:

```bash
poetry install --extras all
poetry run pre-commit install
```

## 🚩 GitHub Issues

Our [issues](https://github.com/vocodedev/vocode-python/issues) page has all of the bugs and enhancements that we want to implement. We have labels that split them into the following categories:

- enhancements: improvements to existing features or altogether new features
- bugs: identified issues that need fixing

If you are working on an issue, please assign to yourself! And please keep issues as modular/focused as possible.

We've marked the issues that we would love folks to work on first with 'good first issue' - see [here](https://github.com/vocodedev/vocode-python/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) for all such issues.

## 🙋Getting Help

We are in the process of making it as easy as possible to contribute but we completely understand if you run into issues.

Please contact us if you need help! We'd love to help + improve the process so that others don't run into similar issues. Come find us on [Discord](https://discord.gg/NaU4mMgcnC)!

We're working on our linting/documentation standards – and will have updates soon. We don't want that to get in your way at all.

## 🏭 Release process

We release updates to Vocode on a daily basis published to [PyPI](https://pypi.org/project/vocode/).

If you submit a PR, we'd love to feature your contribution on Twitter so please include your handle in your PR/otherwise!

## Linting and Typechecking

We use [`black`](https://black.readthedocs.io/en/stable/) for linting. If you're using VSCode, code should auto-format automatically. Otherwise, run the following script before pushing:

```
make lint_diff
```

We use [`mypy`](https://mypy-lang.org/) for typechecking. Full correctness in `vocode/` is required for merging a PR. Run the following script to run this locally:

```
make typecheck # or make typecheck_diff
```

## Testing

We use `pytest` for tests under the `tests` directory. Run tests via:

```
make test
```

## Documentation

We use [Mintlify](https://mintlify.com) for docs, which are found in the `docs` directory. See the [README](https://github.com/vocodedev/vocode-python/blob/main/docs/README.md) for how to update.
