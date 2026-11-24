# Packaged copies of the repo-root configs/ defaults, so that configuration
# loading works from a built wheel (where the repo-root configs/ sibling
# directory does not exist). The repo-root copies remain authoritative for
# research use and take precedence when present; tests/sdk/test_packaging.py
# asserts these copies stay byte-identical, so drift fails CI rather than
# silently changing behavior.
