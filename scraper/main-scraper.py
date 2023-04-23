import re
import requests
from nyt_scraper import scrape_nyt
from politifact_scraper import scrape_politifact
from reuteurs_scraper import scrape_reuteurs
from vokukraine_scraper import scrape_voxukr
from wpost_scraper import scrape_wpost
import pandas as pd
def read_topics(path_to_file):
    topics = []
    with open(path_to_file, "r") as lines:
        for line in lines.readlines():
            line = line.strip()
            line = re.sub(" ", "-", line)
            topics.append(line)
            
    return topics

def main():
    Topics = read_topics(r"C:\Users\karun\Documents\Code\FYP-Fake-News\scraper\topics.txt")
    for main_topic in Topics:
        
        topics = requests.get("https://api.api-ninjas.com/v1/thesaurus?word="+main_topic, headers={"X-Api-Key":"WF/i/5MaoltbNRJ0e6Im1g==Hi6RjZp5V9a2m7mF"}).json()
        for topic in topics["synonyms"] + topics["antonyms"]:
            data = []
            try:
                data.extend(scrape_politifact(topic))
                data.extend(scrape_reuteurs(topic))
                data.extend(scrape_voxukr())
                data.extend(scrape_wpost(topic))
                data.extend(scrape_nyt(topic))
                df = pd.DataFrame(data)
            except Exception as e:
                print(e)
            df = pd.DataFrame(data)
            df.to_csv('./scraper/datasets/'+topic+'.csv')
            print(topic, "done")

main()
