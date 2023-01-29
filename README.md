#  PolitiFact-Scraper - FYP 2023

This Repo comprise a Python web scraper designed to fetch fake checked claims from the fact-checking organization politifact.org. 

## Prerequisites / Requirements

To run the scraper, one must have installed pandas, requests, and beautifulSoup in ones python environment.

There are too simple ways of ensuring you have the requirements.

1) install the requirements to your main python or conda environment.

```bash

pip install -r requirements.txt 

```
2) Create a seperate conda environment for the scraper.

```bash

conda env create -f environment.yml

conda activate politifact-scraper

```


## Running the PolitiFactScraper


```Python

pltscrp = PoltiFactScraper()

pltscrp.full_scrape()

```
