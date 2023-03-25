from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By 

def scrape_voxukr():
    url = "https://voxukraine.org/en/category/voxukraine-informs/"

    options = webdriver.ChromeOptions() 
    options.headless = True 
    with webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options) as driver: 
        driver.get(url)
        data = []
        parent_elements = driver.find_elements(By.CSS_SELECTOR, "body > main > section.base-section.posts-widget > div > div > div:nth-child(1) > article")
        for element in parent_elements:
            claim = element.find_element(By.CSS_SELECTOR, "div.post-info__content > h2").text
            truth_value = "meter-false"
            data.append((claim, truth_value))
        return data
    
