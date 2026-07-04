#!/bin/sh
# Remove Cursor co-author / Made-with trailers from commit messages.
grep -viE '^(co-authored-by:.*cursor|made-with:.*cursor)' || true
