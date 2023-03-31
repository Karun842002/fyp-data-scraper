import re
import pandas as pd
from selenium import webdriver
from selenium.webdriver.edge.options import Options 
from selenium.webdriver.common.by import By 
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import *
import time 
def scrape_polygraph():
    url = f'https://www.polygraph.info/z/7205'.format()

    options = Options() 
    options.add_argument("--headless")
    with webdriver.Edge("/Users/karunanantharaman/Documents/Code/fyp-data-scraper/scraper/msedgedriver", options=options) as driver: 
        driver.get(url)
        data = []
        wait = WebDriverWait(driver, 10, poll_frequency=1, ignored_exceptions=[ElementNotVisibleException, ElementNotSelectableException, InvalidArgumentException])
        wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, '#content > div > div > div > div > div > p > a')))
        button = driver.find_element(By.CSS_SELECTOR, "#content > div > div > div > div > div > p > a")
        for i in range(1000):
            try:
                button.click()
                print("loaded", i)
            except:
                break
            time.sleep(0.5)
        articles = driver.find_elements(By.CSS_SELECTOR, '#blogItems > li')
        for i in range(len(articles)):
            claim = driver.find_element(By.CSS_SELECTOR,f"#blogItems > li:nth-child({i+1}) > div.fc__hdr > a > h4").text.strip()
            truth_value = driver.find_element(By.CSS_SELECTOR, f"#blogItems > li:nth-child({i+1}) > div.fc__body > div.verdict > div.verdict-head > a > span:nth-child(2)").text.strip()
            data.append((claim, truth_value))
        df = pd.DataFrame(data)
        df.to_csv('/Users/karunanantharaman/Documents/Code/fyp-data-scraper/scraper/datasets/polygraph.csv')
        print("done")
        return data

print(len(scrape_polygraph()))