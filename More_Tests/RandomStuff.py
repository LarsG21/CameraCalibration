import cv2
import utils
from datetime import datetime
import csv

print(datetime.now().strftime("%Y-%m-%d"))

with open('Results/eggs.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(['Spam'] * 5 + ['Baked Beans'])
    spamwriter.writerow(['Spam', 'Lovely Spam', 'Wonderful Spam'])


with open('Results/eggs.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
       print(', '.join(row))


starts = [(1,2),(3,4),(5,6)]
ends = [(7,8),(9,10),(11,12)]
distances = [1,7,12]


#utils.writeLinestoCSV(starts,ends,distances)



