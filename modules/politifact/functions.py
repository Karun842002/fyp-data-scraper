import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime

"""
    Code below comprise helper-functions applied when scraping the data from politifact.
    
"""

def get_soup_page(url_page):
    """Converts URL into a BeautifulSoup object.
    Args:
        url_page (_type_): takes a URL page as input parsed as a string.
    Returns:
        _type_: returns a BeautifulSoup object.
    """
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
    exceptions= {  "alcohol":"Alcohol",
                    "candidate-biography": "candidates-biography",
                    "jan.-6": "jan-6",
                    "message-machine-2010":"message-machine",
                    "negative-campaigning": "campaign-advertising",
                    "polls-and-public-opinion": "polls",
                    "race-and-ethnicity": "race-ethnicity",
                    "regulation":"market-regulation",
                    "tampa-bay-10-news": "10-news-tampa-bay"
                }
    with open(path_to_file, "r") as lines:
        for line in lines.readlines():
            line = line.strip()
            line = re.sub(" ", "-", line)
            try:
                line = exceptions[line.lower()]
            except KeyError:    
                line = line.lower()
                
            topics.append(line)
            
    return topics


"""
PREPROCESSING AND FILTERING
Functions below here are related to preprocessing and filtering of the data after it has been fetched. 
All these steps apply to the future process of the Fugazi Project, and can be ignored if you are here just to get the data from politifact


It comprises:
    - Duplicate data removal
    - removing Facebook and Instagram posts
    - sorting and cleaning, keeping the original indexing
"""

def remove_irrelevant_origins(df, undesired_origins=['Facebook posts', 'Instagram posts', 'Viral video']):
    return df[~df['origin'].isin(undesired_origins)]
    

def create_datetime_date_col(df):
    df['date'] = df['stated_on'].apply(lambda x: datetime.strptime(x, '%B %d, %Y'))
    return df

def remove_duplicate_data(df):
    df = create_datetime_date_col(df)
    df = df.sort_values(by='date', ascending=True)
    return df.drop_duplicates(subset='claim', keep='first')


def sort_by_original_index(df, original_index='Unnamed: 0'):
    return df.sort_values(by='Unnamed: 0')


def preprocess_fetched_data(df):
    df = remove_irrelevant_origins(df)    
    df = remove_duplicate_data(df)
    df = sort_by_original_index(df)
    return df


""" Methods below applies to the process extracting the true origins from the URL's that actually reveal the origin
"""

def find_origin_from_url(url):
    pattern = r"\/\d{2}\/"
    match = re.search(pattern, url)
    char_start = match.span()[-1]
    return url[char_start:].split("/")[0]

def split_to_char(word):
    split_word = []
    for s in word:
        split_word.append(s)
    return split_word


def get_jaccard_sim(str1, str2): 
    a = set(split_to_char(str1.lower())) 
    b = set(split_to_char(str2.lower()))
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def jaccard_on_set(set, x):
    most_sim = ("", 0)
    for val in set:
        sim = get_jaccard_sim(val, x)
        if most_sim[-1] < sim:
            most_sim = (val, sim)
            
    return most_sim

