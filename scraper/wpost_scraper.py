import re
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By 
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import *

def scrape_wpost(topic):
    topic = re.sub(" ", "+", topic)
    topic = re.sub("_", "+", topic)
    url = f'https://www.washingtonpost.com/search/?query={topic}&facets=%7B%22time%22%3A%22all%22%2C%22sort%22%3A%22relevancy%22%2C%22section%22%3A%5B%5D%2C%22author%22%3A%5B%5D%7D'.format(topic=topic)

    options = webdriver.ChromeOptions() 
    options.headless = True 
    with webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options) as driver: 
        driver.get(url)
        data = []
        wait = WebDriverWait(driver, 10, poll_frequency=1, ignored_exceptions=[ElementNotVisibleException, ElementNotSelectableException, InvalidArgumentException])
        wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, '#main-content > div.jsx-2309075444.search-app-container.mr-auto.ml-auto.flex.flex-column.col-8-lg > section.jsx-2865089505.search-results-wrapper > button')))
        button = driver.find_element(By.CSS_SELECTOR, "#main-content > div.jsx-2309075444.search-app-container.mr-auto.ml-auto.flex.flex-column.col-8-lg > section.jsx-2865089505.search-results-wrapper > button")
        button.click()
        for article in driver.find_elements(By.CSS_SELECTOR, 'article > div.content-lg.pt-xxxs.pb-xxxs.antialiased.flex.align-items.bc-gray-light > div.pr-sm.flex.flex-column.justify-between.w-100 > div:nth-child(1) > div.font-body.font-light.font-xxs.antialiased.gray-dark.lh-1.mt-sm.mb-sm'):
            claim = article.text.strip()
            truth_value = "meter-true"
            data.append((claim, truth_value))
        return data
