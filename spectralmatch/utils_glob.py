"""Shared wcmatch glob syntax for file searches and image-name rules."""

from wcmatch import glob


# Enable compatible syntax features; retain case sensitivity, native separators, and unique results.
# REALPATH requires existing paths, MATCHBASE implies recursion, and MINUSNEGATE replaces ! exclusions.
GLOB_FLAGS = (
    glob.CASE
    | glob.DOTGLOB
    | glob.EXTGLOB
    | glob.GLOBSTAR
    | glob.GLOBSTARLONG
    | glob.BRACE
    | glob.SPLIT
    | glob.NEGATE
    | glob.NEGATEALL
    | glob.GLOBTILDE
    | glob.RAWCHARS
    | glob.NUMRANGE
)
