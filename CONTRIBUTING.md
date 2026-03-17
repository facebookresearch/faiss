# Contributing to Faiss

We want to make contributing to this project as easy and transparent as
possible.

## Our Development Process

We mainly develop Faiss within Facebook. Sometimes, we will sync the
github version of Faiss with the internal state.

## Pull Requests

We welcome pull requests that add significant value to Faiss. If you plan to do
a major development and contribute it back to Faiss, please contact us first before
putting too much effort into it.

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. If you haven't already, complete the Contributor License Agreement ("CLA").

There is a Facebook internal test suite for Faiss, and we need to run
all changes to Faiss through it.

## Contributor License Agreement ("CLA")

In order to accept your pull request, we need you to submit a CLA. You only need
to do this once to work on any of Facebook's open source projects.

Complete your CLA here: <https://code.facebook.com/cla>

## Issues

We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

Facebook has a [bounty program](https://www.facebook.com/whitehat/) for the safe
disclosure of security bugs. In those cases, please go through the process
outlined on that page and do not file a public issue.

## Coding Style

* 4 spaces for indentation in C++ (no tabs)
* 80 character line length (both for C++ and Python)
* C++ language level: C++17

## Compiler Warnings

Faiss supports configurable compiler warning levels. We encourage contributors
to build with warnings enabled to catch potential issues early.

### Warning Levels

Configure the warning level with `-DFAISS_WARNING_LEVEL=<level>`:

| Level | Description | Recommended For |
|-------|-------------|-----------------|
| 0 | Disabled (default) | Normal builds |
| 1 | Basic warnings (`-Wall -Wextra`) | Development builds |
| 2 | Standard warnings (adds `-Wpedantic`, `-Wshadow`, etc.) | Code review |
| 3 | Strict warnings (all recommended warnings) | Static analysis |

### Example Usage

```bash
# Build with basic warnings
cmake .. -DFAISS_WARNING_LEVEL=1

# Build with warnings as errors (strict mode)
cmake .. -DFAISS_WARNING_LEVEL=2 -DFAISS_WARNINGS_AS_ERRORS=ON
```

### Best Practices for New Code

When writing new code:

* Use `static_cast<>`, `reinterpret_cast<>`, or `const_cast<>` instead of C-style casts
* Initialize all member variables in the member initializer list
* Order member initializers to match declaration order
* Use `[[maybe_unused]]` for intentionally unused parameters
* Use `override` for virtual function overrides
* Prefer `nullptr` over `NULL` or `0` for null pointers
* Avoid shadowing variables in inner scopes

## License

By contributing to Faiss, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
