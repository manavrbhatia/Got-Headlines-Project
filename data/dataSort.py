import csv
import sys

csv.field_size_limit(sys.maxsize)

INPUT_FILE = 'all-years-cnn.csv'
OUTPUT_FILE = '2016-cnn.csv'
PUBLICATION = 'CNN'
BEGIN_YEAR = 2016
END_YEAR = 2016


def read():
	with open(INPUT_FILE, mode='r') as csv_file:
	    csv_reader = csv.DictReader(csv_file)
	    csv_dict = []
	    line_count = 0
	    for row in csv_reader:
	        if line_count != 0:
	        	if row["publication"] == PUBLICATION and int(row["year"]) >= BEGIN_YEAR and int(row["year"]) <= END_YEAR:
	        		csv_dict.append(row)
	        line_count += 1
	    return csv_dict


def write(csv_dict):
	with open(OUTPUT_FILE, mode='w') as csv_file:
	    fieldnames = ['year', 'title', 'article', 'publication']
	    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

	    writer.writeheader()
	    for row in csv_dict:
	    	writer.writerow({'year': row["year"], 'title': row["title"], 'article': row["article"], 'publication': row["publication"]})
	 

dict = read()
write(dict)