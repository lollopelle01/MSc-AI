# Depression analysis on indian students with Bayesian Networks

## Project Structure

```
faikr3-project
├── data_preparation.ipynb      
├── datasets
│   ├── dataset_final.csv       # preparared dataset
│   ├── dataset_raw.csv
│   └── label_mappings.pkl      # to recall the the class encodings
├── network_comparison.ipynb    # (+ queries)
├── papers
|   └── ...                     
├── README.md
├── Report.pdf                  
├── trained_models.pkl          # pretrained models to save time
└── utils.py                    # all complex functions used in the project
```

## Installation
1. Clone this repository:

    ```bash
    git clone https://github.com/Shardyne/faikr3-project.git
    cd ./faikr3-project
    ```

2. Install all required dependencies

    ```bash
    pip3 install -r requirements.txt
    ```