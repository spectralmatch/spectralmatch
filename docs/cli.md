# Command Line Interface

## Installation
The command line interface will be installed automatically when the Python library is installed. See instructions on the installation [page](https://spectralmatch.github.io/spectralmatch/installation/). Use the api reference or command --help to see options to pass into python functions.

## Usage

Print general help:

```bash
spectralmatch --help
```

Print help for a specific command:

```bash
spectralmatch COMMAND --help
```

Print installed version:

```bash
spectralmatch --version
```

Run a specific command:

```bash
spectralmatch COMMAND [OPTIONS]
```

## Pipeline Helper Function
The pipeline function serves as a way to run steps sequentially. While sensible defaults are set, you can modify all params to the underlying functions as seen [here](https://spectralmatch.github.io/spectralmatch/api/pipeline); visit each functions api reference for the exact structure of the params. In its simplest form all that needs to be specified is the input and output:
```commandline
spectralmatch pipeline /input/folder /output/file.tif --shared_debug_logs=True
```

## Commands
{commands_content}