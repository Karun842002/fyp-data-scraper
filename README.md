#  The Epic PolitiFact-Scraper & Datasets

This Repo comprise a Python web scraper designed to fetch fake checked claims from the fact-checking organization politifact.org. Feel free to reach to reach out and use the scraper provided. Make sure to research legal usecases and moral when applying the data.

You are also welcome to reach out to me on Github if you  might have any questions for the scraper.

This repo serves as poart in the Fugazi Project that is Laurenz Aisenpreis and my master thesis project. The Fugazi Project is about understanding the psychological drivers that make people susceptible to sharing and spreading fake news. 


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