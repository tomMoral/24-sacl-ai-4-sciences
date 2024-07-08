# Sacl-AI 2024

## Getting started

### Using `pixi`

An easy way to get started is to use `pixi` for the environment management.
Alternatively, you can use `conda` or `pip` to install the required packages as shown
in the sections below. Refer to [following link](https://pixi.sh/latest/#installation)
for installing `pixi`.

Once `pixi` installed, you just start `jupyterlab`:

```bash
pixi run jupyter lab
```

### Using `conda`

Alternatively you can create a new conda environment which will be called
`sacl-ai` by default and will contain all the packages required to run the
notebooks:

``` bash
conda env create -f environment.yml
```

```bash
conda activate sacl-ai
jupyter notebook  # or jupyter lab
```

You can also update an existing `conda` environment:


``` bash
conda env update -f environment.yml
```

### Using `pip`

We provide a `requirements.txt` file that you can use to install the required
packages using `pip`:

```bash

pip install -r requirements.txt
jupyter notebook  # or jupyter lab
```

## Structure of the repository

A pure python file with `jupytext` markers are available in `python_files` directory.
We converted these files to `ipynb` files using `jupytext` and saved them in the
`notebooks` directory.

Similarly, we provide the corrections of the python files and notebooks in the
`python_files_corrections` and `notebooks_corrections` directories respectively.
