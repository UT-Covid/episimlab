# How to Contribute

First of all, thank you for taking the time to contribute to Episimlab. We've tried to make a stable project and try to fix bugs and add new features continuously. You can help us do more.

## Writing some code!

Contributing to a project on Github is pretty straight forward. If this is you're first time, these are the steps you should take.

1. Fork this repository
1. Add/edit code on your fork 
1. Write a new pytest module for your code and add it to the [tests](./tests) directory
1. Commit and push changes to a branch on your fork
1. Optional: run the pytest suite on your local workstation (see below instructions)
1. Submit a Pull Request against the `main` branch. GitHub Actions will automatically run the pytest suite.

And that's it! Read the code available and change the part you don't like! You're change should not break the existing code and should pass the tests.

When you're done, submit a pull request and for one of the maintainers to check it out. We would let you know if there is any problem or any changes that should be considered.

## Tests

### GitHub Actions 

GitHub Actions CI will automatically run package tests on pull requests and pushes to `main` branch. Therefore, the recommended method of testing is to simply create a new pull request against the `main` branch.

### Testing Locally

Package-level unit tests are written in pytest and we ask that you run them via [tox][1] in a [poetry][2] virtual environment if running locally. To run all package-level unit tests, which require [tox][1] and [poetry][2] installed and in your `$PATH`, issue: `tox`. Tox will pass positional arguments to pytest, e.g. `tox -- --pdb -xk 'test_model_sanity'`

## Documentation

Every chunk of code that may be hard to understand has some comments above it. If you write some new code or change some part of the existing code in a way that it would not be functional without changing it's usages, it needs to be documented.

## References

Credit to [`mpourismaiel`'s gist](https://gist.github.com/mpourismaiel/6a9eb6c69b5357d8bcc0) for this document's template.

[1]: https://tox.readthedocs.io
[2]: https://python-poetry.org
