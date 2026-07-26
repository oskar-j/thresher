"""Message templates for the errors this package raises.

Kept as format strings in one place so the wording stays consistent, and so tests can
assert against the same text the user sees.
"""

NOT_IMPLEMENTED_ERROR = 'The "scores" attribute is not an Iterable! Please provide a list-like object'
UNKNOWN_ALGORITHM = "Unknown algorithm. Run get_supported_algorithms() to get a list of available algorithms."

UNKNOWN_ALGORITHM_NAME = (
    "Unknown algorithm {name!r}. Available algorithms are: {available}. "
    "Run get_supported_algorithms() to list them at runtime."
)

UNEXPECTED_LABELS = (
    'Found {unexpected} in "actual_classes", but only -1 and 1 are supported. '
    'If your data uses different labels, declare them with the "labels" option, '
    "for example Thresher(labels=(0, 1))."
)

SINGLE_CLASS_LABELS = (
    '"actual_classes" contains only {only}. Both -1 and 1 must be present - '
    "a threshold cannot be optimized against a single class."
)

LENGTH_MISMATCH = (
    'Got {scores} scores but {classes} entries in "actual_classes". Each score needs the '
    "class it belongs to, so the two must be the same length."
)

MISSING_LABELS = (
    '"actual_classes" contains {count} missing value(s). Every score needs a known class; '
    "rows with a blank or NaN label have to be filled in or dropped before optimizing."
)

EMPTY_INPUT = '"scores" and "actual_classes" are empty - there is nothing to optimize.'
