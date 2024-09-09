import csv
import os
from io import open
import codecs


def readPairFromFile(filename):
    qa_pairs = []
    with open(filename, 'r') as datafile:
        lines = datafile.readlines()
    for i in range(0, len(lines) - 1):
        qa_pairs.append([lines[i].strip(), lines[i + 1].strip()])
    return qa_pairs


infile = "data/movie-corpus/Borderlands.txt"
outfile = "data/movie-corpus/out.txt"

delimiter = '\t'
delimiter = str(codecs.decode(delimiter, "unicode_escape"))

with open(outfile, 'w', encoding='utf-8') as outputfile:
    writer = csv.writer(outputfile, delimiter=delimiter, lineterminator='\n')
    for pair in readPairFromFile(infile):
        writer.writerow(pair)