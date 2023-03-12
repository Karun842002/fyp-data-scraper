# from deep_translator import GoogleTranslator
import json

from deep_translator import GoogleTranslator
from tqdm import tqdm

with open('/Users/karunanantharaman/Documents/Code/fyp-data-scraper/datasets/gfcmerged.json') as file:
    content = file.read()
    object = json.loads(content)
    for idx in tqdm(range(len(object))):
        object[idx]["text"] = GoogleTranslator(source='auto', target='en').translate(object[idx]["text"])
        object[idx]["claimReview"][0]["textualRating"] = GoogleTranslator(source='auto', target='en').translate(object[idx]["claimReview"][0]["textualRating"])
        # print(f"{idx} of {len(object)}", object[idx]["text"])
    outfile = open('/Users/karunanantharaman/Documents/Code/fyp-data-scraper/datasets/gfcmergednew.json', "w")
    json.dump(object, outfile)
    