"""
@Gyrst - created in September 2022

When running the politifact scraper make sure to:
1) specify and check that the path for the destination directory is correct.
2) Add a title descriptive of what has been fetched.
3) change "topics" to the desired topics that you would like to fetch. You can add or remove topics to the topcs.txt file
"""

import re
import pandas as pd
from functions import *
from datetime import date

#Remember to specify title to avoid overwriting existing files
title = "_alltopics_"
original_topics = []
topics = read_topics("/Users/karunanantharaman/Documents/Code/politifact-fakenews-scraper/modules/politifact/topics.txt")
data = []

for topic in topics:
    
    #In order to avoid scraping the topics already scraped earlier.
    if topic in original_topics:
        continue
    
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
                
                #retrieve the claim
                claim = lst_items[i].find_all('a')[1].get_text().strip()

                #retrieve url
                url_ending = str(lst_items[i].find_all('a', href=True)[1]).split("\">")[0].split("<a href=\"")[1]
                url = "https://www.politifact.com{url_ending}".format(url_ending=url_ending)

                #retrieve truth value (e.g., "barely-true")
                truth_value = str(lst_items[i].find_all(attrs={'m-result__media'})).split("images/politifact/rulings/")[1].split("/")[0]
                
                #retrieve the origin of the claim (e.g., instagram posts, fox news, joe biden)
                origin = soup_object.find(attrs={"c-textgroup__author"}).find('a', href=True).get_text().strip().splitlines()[0]
                
                #retrieve date for when the fake news started spreading
                date_raw = lst_items[i].find_all(attrs={"c-textgroup__author"})[0].get_text().strip()
                stated_on = re.search("([^\s]+)\s+\d{1,2}.\s20\d\d", date_raw)[0]
                
                
                news_article = [claim, origin, url, truth_value, stated_on, topic]
                
                # printing the first set from each iteration in the output to stay informed on the progress made
                
                data.append(news_article)
                
        except Exception as e:
            print("Error occured at", page_number, base_url, e)
        page_number += 1
        
    
    title = topic
    df = pd.DataFrame(data)
    print(df)
    df = df.rename({df.columns[0]:'claim', df.columns[1]:'origin', df.columns[2]:'URL', df.columns[3]:'truth_value', df.columns[4]:'stated_on',df.columns[5]:'topic'}, axis=1)

    path = "data/politifact/"
    csv_title = "politifact_scrape{title}{date}.csv".format(title=title, date=str(date.today().strftime("%d%m%Y")))


    print(csv_title)
    df.to_csv(path+csv_title)
    df.to_csv(csv_title)
