from functions import *
import pandas as pd
import os, json

url = "https://www.politifact.com/personalities/"

soup_object = get_soup_page(url)

origin_party_dict = {}

save = True
merge_csv = False

#the class tags in the html
abc_class_title = "o-platform o-platform--has-thin-border o-platform--is-wide"
platform_content = 'o-platform__content'
tuple_val = 'c-chyron'
person = 'c-chyron__value'
party = 'c-chyron__subline'


# Lst of all the all personalities by letter a,b,c,d etc...
lst_abc = soup_object.find_all(attrs={'o-platform o-platform--has-thin-border o-platform--is-wide'})

for val in lst_abc:
    content = val.find_all(attrs={platform_content})
    pairs = content[0].find_all(attrs={tuple_val})
    for pair in pairs:
        origin = pair.getText().strip().split("\n")[0]
        party = pair.getText().strip().split("\n")[-1]
        origin_party_dict[origin] = party



#to dataframe
origins_pol = pd.DataFrame([origin_party_dict.keys(), origin_party_dict.values()]).transpose()
origins_pol = origins_pol.set_index(0)

#join the two
pols = pd.read_csv(os.getcwd() + "/data/politifact/all_politifact_1210nodup_fromnotopdup.csv")
pols = pols.join(origins_pol, on='origin', how='left')




if save:
    with open("origins_political_affiliation.json", "w") as fp:
        json.dump(origin_party_dict, fp)
        

if merge_csv:
    pols.to_csv("0411_pltfact_w_origin_poltag.csv")