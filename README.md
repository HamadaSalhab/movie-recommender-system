# Practical Machine Learning and Deep Learning - Assignment 2 - Movie Recommender System

Innopolis University

Fall 2023

Practical Machine Learning and Deep Learning

Student: Hamada Salhab

Email: <h.salhab@innopolis.university>

Group: BS21-AI

## Prerequisites

- Python (preferrably Python 3.8).
- Git.
- Anaconda or Miniconda.

## How to Use

## Getting Started

1- Clone the repository. You can do it through the following command in a terminal window:

``` zsh
git clone https://github.com/HamadaSalhab/movie-recommender-system
```

2- Navigate to the repo's root directory:

``` zsh
cd movie-recommender-system
```

3- Set up the environment. You can create a virtual environment using the following commands:

  1. Create a new conda environment (you can change the environment's name from myconda to whatever you want):

``` zsh
conda create --name myconda
```

  2. Run the following command to activate the environment you've just made:

```zsh
conda activate myconda
```

4- Install the required Python dependencies:

``` zsh
conda install -c conda-forge --file requirements.txt
```

### Get Data

To download the [MovieLens 100K dataset](https://grouplens.org/datasets/movielens/100k/) dataset, run the following command from the repo's root directory:

``` zsh
python3 src/data/make_dataset.py
```

This will save the raw data in the following path:

```directory
movie-recommender-system
└───data
    └───raw
        └───ml-100k
            └───ml-100k
```

### Explore & Transform Data

To explore and pre-process the data, you can run the "1.0-initial-data-exploration.ipynb" Jupyter Notebook. The notebook can be found in the following path:

```directory
movie-recommender-system
└───notebooks
    └───1.0-initial-data-exploration.ipynb
```

### Train

To train the LightFM model, you can run the "2.2-model-training-LightFM.ipynb" Jupyter Notebook up until the "Create & fit the model with best parameters" sub-section. The notebook can be found in the following path:

```directory
movie-recommender-system
└───notebooks
    └───2.2-model-training-LightFM.ipynb
```

### Predict

To make predictions using the LightFM model, you should run the following sections in the "2.2-model-training-LightFM.ipynb" Jupyter Notebook:

1. First, start with the "Prepare the Dataset" Section. It has 8 cells.
2. Then, skip the "Train LightFM Model" section and run the "Load" section from the "Save & Load the model checkpoint".
3. Optionally, you can run the "Evaluate" section to evaluate the model on the test dataset.
4. Finally, run the "Make Recommendation" section.

The notebook can be found in the following path:

```directory
movie-recommender-system
└───notebooks
    └───2.2-model-training-LightFM.ipynb
```

### Evaluate

To evaluate the LightFM model on the evaluation dataset, run the following command from the root directory of this repo:

``` zsh
python3 benchmark/evaluate.py
```

## References

1. McKinney, W. et al. (2023). pandas. Retrieved from pandas.pydata.org.
2. Hunter, J. D. et al. (2023). Matplotlib. Retrieved from matplotlib.org​​.
3. Waskom, M. et al. (2021). seaborn. Retrieved from seaborn.pydata.org​​.
4. Harris, C.R. et al. (2022). NumPy. Retrieved from numpy.org​​.
5. Hug, N. (2023). Surprise: A Python scikit for recommender systems. Retrieved from surpriselib.com​​.
6. Kula, M. (2023). LightFM. Retrieved from LightFM documentation​​.
