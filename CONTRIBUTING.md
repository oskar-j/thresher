# Contributing

## Co-ordinator

Oskar Jarczyk (`oskar.jarczyk@gmail.com`)

## Adding features or fixing bugs

* Fork the repo
* Check out a feature or bug branch
* Add your changes
* Update README when needed
* Submit a pull request to upstream repo
* Add description of your changes
* Ensure tests are passing
* Ensure branch is mergeable

## Testing

* Set up the environment with `uv sync --group dev`
* Please make sure the full suite passes: `uv run pytest`
* Run the checks CI runs before pushing: `uv run pre-commit run --all-files`
* Optionally `uv run pre-commit install` so those checks run on every commit

Pull requests are gated on both: the `pre-commit` job and the `test` matrix must be green
before a branch can be merged into `main`.

## What is wanted at the moment

* Check "Issues" section for some job for you
* Spread some word on social media about this package
* Stay positive :)
