# Tensorboard Logging

This package provides a simple logger class that supports logging scalars and images to tensorboard without using tensorflow.

## Usage

It takes a few steps to setup a logger.

- Create a logger specifying the `logdir` and `subfolder`. The summaries will be saved 
under `logdir/time stamp/subfolder`.
- Register scalars and images to be logged with `tag` and `categoty`. The `tag` should be 
unique.
- Freeze the logger.

To do logging, just feed the global `step` and the dict of `tag`s and values.