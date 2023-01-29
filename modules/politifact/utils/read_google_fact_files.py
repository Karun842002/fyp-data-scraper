import json, os, pandas as pd

#Google Fact Checking API
#https://developers.google.com/fact-check/tools/api/reference/rest/v1alpha1/claims/search?apix_params=%7B%22languageCode%22%3A%22Python%22%2C%22maxAgeDays%22%3A900%2C%22pageSize%22%3A100%2C%22query%22%3A%22climate%20chnage%22%7D#request-body

#requests.get('https://factchecktools.googleapis.com/v1alpha1/claims:search') #needs API KEY ETC!


#preprocessing the data into a csv file

project_path = os.getcwd()
files = os.listdir(project_path + "/data/google_fact_check/")

print(project_path)
print(files)

def expand_value(x, value):
    try:
        return x.claimReview[0][value]
    except:
        return None

def get_topic(file):
    return file.split("gfc_")[1].split("5000")[0]


lst = []

for file in files:
    with open(project_path+"/data/google_fact_check/"+file) as f:
        result = json.load(f)
    
    df =  pd.DataFrame(result['claims'])
    
    lst.append(df)
    

df = pd.concat(lst)


extra_details = ['publisher', 'truth_value', 'textualRating', 'url', 'title']

df['fact_checker'] = df.apply(lambda x: x.claimReview[0]['publisher'], axis=1)
df['rating'] = df.apply(lambda x: x.claimReview[0]['textualRating'], axis=1)
df['fact_checker_url'] = df.apply(lambda x: x.claimReview[0]['url'],  axis=1)
df['title_factcheck_page'] = df.apply(lambda x: expand_value(x, 'title'), axis=1)

df.to_csv(project_path+"/data/google_fact_check/gfc.csv")
