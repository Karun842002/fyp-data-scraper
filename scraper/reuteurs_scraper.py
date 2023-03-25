import re
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By 

def scrape_reuteurs(topic):
    topic = re.sub(" ", "+", topic)
    topic = re.sub("_", "+", topic)
    url = "https://www.reuters.com/site-search/?query={topic}&offset=0&section=world&sort=newest".format(
        topic=topic.replace)

    options = webdriver.ChromeOptions() 
    options.headless = True 
    with webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options) as driver: 
        driver.get(url)
        data = []
        parent_elements = driver.find_elements(By.CSS_SELECTOR, "#fusion-app > div.search-layout__body__1FDkI > div.search-layout__main__L267c > div > div:nth-child(3) > div.search-results__sectionContainer__34n_c > ul > li")
        for element in parent_elements:
            claim = element.find_element(By.CSS_SELECTOR, "li > div > div.media-story-card__body__3tRWy > a > span:nth-child(1)").text
            truth_value = "meter-true"
            data.append((claim, truth_value))
        return data
