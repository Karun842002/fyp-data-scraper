import re
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By 
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import *

def scrape_nyt(topic):
    topic = re.sub(" ", "%20", topic)
    topic = re.sub("_", "%20", topic)
    url = f'https://www.nytimes.com/search?dropmab=false&endDate=20230227&query={topic}&sort=newest&startDate=20220227'.format(topic=topic)

    options = webdriver.ChromeOptions() 
    options.headless = True 
    with webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options) as driver: 
        driver.get(url)
        data = []
        wait = WebDriverWait(driver, 10, poll_frequency=1, ignored_exceptions=[ElementNotVisibleException, ElementNotSelectableException, InvalidArgumentException])
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, '#site-content > div > div:nth-child(2) > div.css-46b038 > ol')))
        for article in driver.find_elements(By.CSS_SELECTOR, '#site-content > div > div > div.css-46b038 > ol > li > div > div > div > a > p.css-16nhkrn'):
            claim = article.text.strip()
            truth_value = "meter-true"
            data.append((claim, truth_value))
        return data
    

