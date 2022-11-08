import csv
import sys
from constants import INPUT_FILE

csv.field_size_limit(sys.maxsize)

def read(begin_year=2016, end_year=2016, publications=["CNN", "Reuters", "The New York Times"]):
	"""
	Parse and extract desired labels in All The News

	Arguments:
	begin_year: First year you want to include
	end_year: Last year you want to include
	publications: All publications you want to include
	"""

	with open(INPUT_FILE, mode='r') as csv_file:
	    csv_reader = csv.DictReader(csv_file)
	    csv_dict = []
	    line_count = 0
	    for row in csv_reader:
	        if line_count != 0:
	        	if row["publication"] in publications and int(row["year"]) >= begin_year and int(row["year"]) <= end_year:
	        		csv_dict.append(row)
	        line_count += 1
	    return csv_dict

def write(csv_dict, out_file : str):
	"""
	Write the parsed file you want to parse and modify.

	Arguments:
	csv_dict: Parsed and modified file you want to output
	out_file: Name of your new CSV
	"""

	with open(out_file, mode='w') as csv_file:
	    fieldnames = ['year', 'title', 'article', 'publication']
	    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

	    writer.writeheader()
	    for row in csv_dict:
	    	writer.writerow({'year': row["year"], 'title': row["title"], 'article': row["article"], 'publication': row["publication"]})
	 

def select_dataset(out_file : str, begin_year=2016, end_year=2016, publications=["CNN", "Reuters", "The New York Times"]):
	"""
	Specify labels you want to extract from All The News 2.0 and output a new CSV using those

	Arguments:
	out_file (str): Name of your new CSV
	begin_year (int): First year you want to include
	end_year (int): Last year you want to include
	publications (List(str)): List of publications you want to include
	"""

	dict = read()
	write(dict, out_file)