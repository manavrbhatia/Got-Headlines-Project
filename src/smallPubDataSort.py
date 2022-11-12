import csv
import sys
from constants import INPUT_FILE
from dataSort import select_dataset

csv.field_size_limit(sys.maxsize)


publications = ["Axios", "Business Insider", "Buzzfeed News", "CNBC", "CNN", "Economist", 
"Fox News", "Gizmodo", "Hyperallergic", "Mashable", "New Republic", "New Yorker", "People",	
"Politico", "Refinery 29", "Reuters", "TMZ", "TechCrunch", "The Hill", "The New York Times",
"The Verge", "Vice", "Vice News", "Vox", "Washington Post", "Wired"]


def generate():
    for pub in publications:
        select_dataset("../data/"+pub+".csv", 2016, 2020, [pub])
    
    
