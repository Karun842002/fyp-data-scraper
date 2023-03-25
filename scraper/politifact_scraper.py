import re
from bs4 import BeautifulSoup
import requests

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
                truth_value = str(lst_items[i].find_all(attrs={'m-result__media'})).split("images/politifact/rulings/")[1].split("/")[0]
                data.append((claim, truth_value))  
            break   
        except Exception as e:
            print("Error occured at", page_number, base_url, e)
        page_number += 1
    return data