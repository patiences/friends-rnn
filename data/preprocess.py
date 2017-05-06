import os

filenames = []
for filename in os.listdir("data/"):
    if filename.endswith(".txt"):
        filenames.append("data/" + filename)

with open("input.txt", "a") as outfile:
    for fname in filenames:
        print(fname)
        with open(fname) as infile:
            for line in infile:
                outfile.write(line)
