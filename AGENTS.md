This project is for modeling discrete 2D images of land cover / land use data using the Random order autoregressive GPT model.

- Make sure to read the README file to understand this project.
- Before proceeding further, also read the `train.py` file as well as the code for `RandAR / model / randar_gpt.py`.
- For the theoretical underpinnings of the main model in this work, look at the file `.research` for a LaTex document outlining the original RandAR model

### Style
When writing scripts (e.g. files focused around 1 or a few outputs) make the style **flat** and **concise**. ONLY write functions if they will be frequently reused. ALWAYS specify parameters or config values in a `dataclass` at the start of the file. Assume that the code will not be a command line utility UNLESS specifically asked to do so. For utilities, write them as modules then import them into `tasks.py` and run them using Invoke like `inv check-status`.

Write code in a minimal, compact fashion.

