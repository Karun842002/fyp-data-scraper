import re
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By 
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import *
import time 
def scrape_polygraph():
    url = f'https://www.polygraph.info/z/7205'.format()

    options = webdriver.ChromeOptions() 
    # options.headless = True 
    with webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options) as driver: 
        driver.get(url)
        data = []
        wait = WebDriverWait(driver, 10, poll_frequency=1, ignored_exceptions=[ElementNotVisibleException, ElementNotSelectableException, InvalidArgumentException])
        wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, '#content > div > div > div > div > div > p > a')))
        button = driver.find_element(By.CSS_SELECTOR, "#content > div > div > div > div > div > p > a")
        for i in range(1000):
            button.click()
            time.sleep(1)
        articles = driver.find_elements(By.CSS_SELECTOR, '#blogItems > li')
        for i in range(len(articles)):
            claim = driver.find_element(By.CSS_SELECTOR,f"#blogItems > li:nth-child({i+1}) > div.fc__hdr > a > h4").text.strip()
            truth_value = driver.find_element(By.CSS_SELECTOR, f"#blogItems > li:nth-child({i+1}) > div.fc__body > div.verdict > div.verdict-head > a > span:nth-child(2)")
            data.append((claim, truth_value))
        return data

print(len(scrape_polygraph))