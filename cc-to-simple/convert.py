import csv

with open('D:\\FYP\\fyp-data-scraper\\datasets\\cleaned1.csv') as file_obj:
    reader_obj = csv.reader(file_obj)
    wf = open('./datasets/cleaned1.txt', 'w')
    for row in reader_obj:
        wf.write(row[0] + ' .\n')
