## Glob patterns

File searches and image-name rules use [wcmatch.glob](https://facelessuser.github.io/wcmatch/glob/) with all compatible syntax features enabled: `*`, `?`, character classes (including POSIX classes), braces, extglobs, numeric ranges, `|` alternatives, `!` exclusions, tilde expansion, character escapes, hidden names, and recursive wildcards. Matching is case-sensitive, duplicate results are removed, and pattern expansion has no count limit.

| Pattern | Meaning |
| --- | --- |
| `input/*.{tif,tiff}` | Both raster extensions |
| `input/@(base\|scene_*).tif` | Extended alternatives within a filename |
| `input/scene_<1-20>.tif` | Numeric range |
| `input/*.tif\|!input/bad*` | TIFF files excluding names starting with `bad` |
| `input/**/*.tif` | All depths, including hidden folders |
| `input/***/*.tif` | All depths, also traversing directory symlinks |
| `~/imagery/*.tif` | Files under the user's home directory |

`search_paths` enables recursive wildcards by default; `recursive=False` treats `**` and `***` as ordinary `*`. A pattern without recursive wildcards searches only its specified levels. Folder inputs require `default_file_pattern`, which is evaluated relative to that folder, including alternatives and exclusions. Lists of input paths remain literal paths. Escape literal pattern characters with a backslash. Use `/` as a path separator on all platforms.

::: spectralmatch.handlers
