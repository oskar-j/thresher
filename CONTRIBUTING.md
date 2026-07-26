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

* Set up with `make install`, which installs the dependencies and the git pre-commit hook
* Please make sure the full suite passes: `make test`
* Run everything CI runs before pushing: `make check`
* Coverage must stay at or above 90%: `make cov`. CI enforces this on every supported
  Python version, so a pull request that drops below it cannot be merged

Prefer `make check` over calling `pre-commit run --all-files` yourself. That command only
sees files git already tracks, so a file you have just created is skipped without comment
and the run reports success having never looked at it - CI then fails on the very thing
your green local run missed. `make check` registers new paths first so the hooks see them.
`make help` lists the rest.

Pull requests are gated on both: the `pre-commit` job and the `test` matrix must be green
before a branch can be merged into `main`.

## What is wanted at the moment

* Check "Issues" section for some job for you
* Spread some word on social media about this package
* Stay positive :)
