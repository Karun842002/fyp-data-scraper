import re
import pandas as pd
from selenium import webdriver
from selenium.webdriver.edge.options import Options 
from selenium.webdriver.common.by import By 
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import *
import time 
def scrape_disinfo():
    data = []
    for j in range(300):
        url = f'https://euvsdisinfo.eu/disinformation-cases/?disinfo_keywords%5B0%5D=keyword_77110&date=&offset={j*10}&per_page=10'
        options = Options() 
        options.add_argument("--headless")
        with webdriver.Edge("/Users/karunanantharaman/Documents/Code/fyp-data-scraper/scraper/msedgedriver", options=options) as driver: 
            driver.get(url)
            wait = WebDriverWait(driver, 10, poll_frequency=1, ignored_exceptions=[ElementNotVisibleException, ElementNotSelectableException, InvalidArgumentException, TimeoutException])
            wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, '#main-content > div > div > div > div.js-content > table > tbody > tr')))
            articles = driver.find_elements(By.CSS_SELECTOR, '#main-content > div > div > div > div.js-content > table > tbody > tr')
            for i in range(len(articles)):
                try:
                    claim = driver.find_element(By.CSS_SELECTOR,f"#main-content > div > div > div > div.js-content > table > tbody > tr:nth-child({i+1}) > td.disinfo-db-cell.cell-title > a").text.strip()
                    truth_value = 'meter-false'
                    data.append((claim, truth_value))
                except:
                    continue
        print("done page", j)
    df = pd.DataFrame(data)
    df.to_csv('/Users/karunanantharaman/Documents/Code/fyp-data-scraper/scraper/datasets/disinfo.csv')
    print("done")
    return data

print(len(scrape_disinfo()))