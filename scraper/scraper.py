import asyncio
import re
import pandas as pd
from datetime import date
import requests
import os
from bs4 import BeautifulSoup
from pyppeteer import launch
from dotenv import load_dotenv
import time
load_dotenv()

def get_soup_page(url_page):
    response = requests.get(url_page)
    page = BeautifulSoup(response.content, 'html.parser')
    return page

def more_pages(soup_object):
    try:
        if soup_object.find(attrs="c-title c-title--subline").get_text().strip()=='No Results found':
            return False
        else:
            return True
    except:
        return True

def read_topics(path_to_file):
    topics = []
    with open(path_to_file, "r") as lines:
        for line in lines.readlines():
            line = line.strip()
            line = re.sub(" ", "-", line)
            topics.append(line)
            
    return topics

def delay(time):
    return asyncio.sleep(time)

def scrape_politifact(topic):
    data = []
    page_number = 1
    base_url = 'https://www.politifact.com/search/factcheck/?page={page_number}&q={topic}'.format(page_number=page_number, topic=topic)
    print("Currently running for", topic)
    while more_pages(get_soup_page(base_url)):
        try:
            base_url = 'https://www.politifact.com/search/factcheck/?page={page_number}&q={topic}'.format(page_number=page_number, topic=topic)
            soup_object = get_soup_page(base_url)
            def is_item(x):
                if not x.has_attr('class'):
                    return False
                if "m-result" in x['class']:
                    return True
                return False 
            lst_items = soup_object.find_all(is_item)
            if len(lst_items)==0:
                print("hit page limit at:", page_number, "for topic:", topic)
                break
            for i in range(len(lst_items)):
                claim = lst_items[i].find_all('a')[1].get_text().strip()
                url_ending = str(lst_items[i].find_all('a', href=True)[1]).split("\">")[0].split("<a href=\"")[1]
                url = "https://www.politifact.com{url_ending}".format(url_ending=url_ending)
                truth_value = str(lst_items[i].find_all(attrs={'m-result__media'})).split("images/politifact/rulings/")[1].split("/")[0]
                origin = soup_object.find(attrs={"c-textgroup__author"}).find('a', href=True).get_text().strip().splitlines()[0]
                date_raw = lst_items[i].find_all(attrs={"c-textgroup__author"})[0].get_text().strip()
                stated_on = re.search("([^\s]+)\s+\d{1,2}.\s20\d\d", date_raw)[0]
                news_article = [claim, origin, url, truth_value, stated_on, topic]
                data.append(news_article)     
        except Exception as e:
            print("Error occured at", page_number, base_url, e)
        page_number += 1
    return data
        
async def scrape_reuters(topic):
    base_url = "https://www.reuters.com/site-search/?query={topic}&offset=0&section=world&sort=newest".format(topic=topic)
    browser = await launch(headless=True)
    page = await browser.newPage()
    await page.goto(base_url, {'waitUntil': 'networkidle2'})
    results = []
    while True:
        await page.waitForSelector('.search-results__list__2SxSK')
        html = await page.content()
        soup = BeautifulSoup(html, 'html.parser')

        news = soup.select(
            '#fusion-app > div > div.search-layout__body__1FDkI > div.search-layout__main__L267c > div > div:nth-child(3) > div.search-results__sectionContainer__34n_c > ul > li')
        
        for i in range(len(news)):
            claim = soup.select(f'#fusion-app > div > div.search-layout__body__1FDkI > div.search-layout__main__L267c > div > div:nth-child(3) > div.search-results__sectionContainer__34n_c > ul > li:nth-child({i+1}) > div > div.media-story-card__body__3tRWy > a')[0].text
            date_of_publish = soup.select(f'#fusion-app > div > div.search-layout__body__1FDkI > div.search-layout__main__L267c > div > div:nth-child(3) > div.search-results__sectionContainer__34n_c > ul > li:nth-child({i+1}) > div > div.media-story-card__body__3tRWy > time')[0].text
            results.append({'claim': claim, 'dateOfPublish': date_of_publish})

        page_count = soup.select('#fusion-app > div > div.search-layout__body__1FDkI > div.search-layout__main__L267c > div > div:nth-child(3) > div.search-results__pagination__2h60k > span')[0].text
        browsed_pages = int(page_count.split('to')[1])
        total_pages = int(page_count.split('of')[1])

        if browsed_pages == total_pages:
            break
        
        next_button = await page.select_one('#fusion-app > div > div.search-layout__body__1FDkI > div.search-layout__main__L267c > div > div:nth-child(3) > div.search-results__pagination__2h60k > button:nth-child(3)')
        await next_button.click()
        await delay(2)

    await browser.close()
    return results
    
async def scrape_voxukr():
    base_url = "https://voxukraine.org/en/category/voxukraine-informs/"
    browser = await launch(headless=True)
    page = await browser.newPage()
    await page.goto(base_url, {'waitUntil': 'networkidle2'})
    results = []
    for i in range(1, 101):
        try:
            soup = BeautifulSoup(page.content(), 'html.parser')

            news = soup.select(f'body > main > section.base-section.posts-widget > div > div > div:nth-child({i}) > article')

            for article in news:
                claim = article.select_one('div.post-info__content > h2').get_text(strip=True)
                dateOfPublish = article.select_one('div.post-info__content > div.post-info__date').get_text(strip=True)
                results.append({'claim': claim, 'dateOfPublish': dateOfPublish})

            nextButton = await page.querySelector(f'body > main > section.base-section.posts-widget > div > div > div:nth-child({i+1}) > button')
            await nextButton.click()
            await delay(2)
        except Exception as err:
            print(err)
            break

    await browser.close()
    return results    
    
async def get_html(page):
    body = await page.xpath('//body')
    html = await page.evaluate('(body) => body.innerHTML', body[0])
    return html

async def scrape_wpost(topic):
    base_url = f'https://www.washingtonpost.com/search/?query={topic}&facets=%7B%22time%22%3A%22all%22%2C%22sort%22%3A%22relevancy%22%2C%22section%22%3A%5B%5D%2C%22author%22%3A%5B%5D%7D'.format(topic=topic)
    browser = await launch(headless=True)
    page = browser.newPage()
    await page.goto(base_url, {'waitUntil': 'networkidle2'})
    articles = []
    for i in range(1000):
        soup = BeautifulSoup(await get_html(page), 'html.parser')
        try:
            button = await page.waitForSelector('#main-content > div.jsx-2309075444.search-app-container.mr-auto.ml-auto.flex.flex-column.col-8-lg > section.jsx-2865089505.search-results-wrapper > button')
            await button.click()
            await delay(1)
        except:
            break
    soup = BeautifulSoup(get_html(page), 'html.parser')
    for article in soup.select('article > div.content-lg.pt-xxxs.pb-xxxs.antialiased.flex.align-items.bc-gray-light > div.pr-sm.flex.flex-column.justify-between.w-100 > div:nth-child(1) > div.font-body.font-light.font-xxs.antialiased.gray-dark.lh-1.mt-sm.mb-sm'):
        claim = article.text.strip()
        articles.append({'id': i, 'claim': claim, 'truth_value': 'meter-true'})
    for i, title in enumerate(soup.select('article > div.content-lg.pt-xxxs.pb-xxxs.antialiased.flex.align-items.bc-gray-light > div.pr-sm.flex.flex-column.justify-between.w-100 > div:nth-child(1) > a')):
        articles[i]['title'] = title.text.strip()
    for i, stated_on in enumerate(soup.select('article > div.content-lg.pt-xxxs.pb-xxxs.antialiased.flex.align-items.bc-gray-light > div.pr-sm.flex.flex-column.justify-between.w-100 > div:nth-child(2) > span')):
        articles[i]['stated_on'] = stated_on.text.strip()
    await browser.close()
    return articles
        

async def main():
    topics = read_topics("/workspaces/fyp-data-scraper/scraper/topics.txt")
    for main_topic in topics:
        data = []
        topics = requests.get("https://api.api-ninjas.com/v1/thesaurus?word="+main_topic, headers={"X-Api-Key":os.getenv('API_KEY')}).json()
        for topic in topics["synonyms"] + topics["antonyms"]:
            data.extend(scrape_politifact(topic))
            data.extend(await scrape_reuters(topic))
            data.extend(await scrape_voxukr())
            data.extend(await scrape_wpost(topic))
            
asyncio.get_event_loop().run_until_complete(main())