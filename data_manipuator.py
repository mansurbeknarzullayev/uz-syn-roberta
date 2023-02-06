import pandas as pd

dataset_file = 'data/dataset.txt'
wfile = open(dataset_file, 'w')
wfile.write('')
wfile.close()

def readContent(file):
    wfile = open(dataset_file, 'a')
    df = pd.read_csv('data/' + file)
    for line in df['content']:
        line = str(line).strip()
        line = line.replace(' ', ' ')
        if line != '':
            wfile.write(line + '\n')
    wfile.close()

files = ['kun.uz.csv', 'daryo.uz.csv']

for file in files:
    readContent(file)