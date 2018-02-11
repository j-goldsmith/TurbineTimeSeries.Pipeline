import csv


def _load_config(file_path):
    data = {}
    with open(file_path) as f:
        reader = csv.reader(f, skipinitialspace=True, quotechar="'")
        for row in reader:
            data[row[0]] = row[1]

    return data
