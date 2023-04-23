import re
import pandas as pd
from selenium import webdriver
from selenium.webdriver.edge.options import Options 
from selenium.webdriver.common.by import By 
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import *
import time 
def scrape_stopfake():
    data = []
    for j in range(1,200):
        try:
            url = f'https://www.stopfake.org/ru/category/factcheck_for_facebook_ru/page/{j}/'
            options = Options() 
            options.add_argument("--headless")
            with webdriver.Edge("/Users/karunanantharaman/Documents/Code/fyp-data-scraper/scraper/msedgedriver", options=options) as driver: 
                driver.get(url)
                articles = driver.find_elements(By.CSS_SELECTOR, '#td-outer-wrap > div.td-main-content-wrap.td-container-wrap > div > div > div.td-pb-span8.td-main-content > div > div')
                for i in range(len(articles)):
                    try:
                        claim = driver.find_element(By.CSS_SELECTOR,f"#td-outer-wrap > div.td-main-content-wrap.td-container-wrap > div > div > div.td-pb-span8.td-main-content > div > div:nth-child({i+1}) > div.item-details > h3 > a").text.strip()
                        truth_value = 'meter-false'
                        data.append((claim, truth_value))
                    except:
                        continue
            print("done page", j)
        except:
            continue
    df = pd.DataFrame(data)
    df.to_csv('/Users/karunanantharaman/Documents/Code/fyp-data-scraper/scraper/datasets/stopfake.csv')
    print("done")
    return data

print(len(scrape_stopfake()))