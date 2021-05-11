import sys

modelFile = sys.argv[1]
outFileName = sys.argv[2]
outFile = open(sys.argv[2], 'w')
nbwords = 0
dictW = {}
freqs = []
cleanWord = ""
rank = 1

# this function removes opening apostrophes from words, and closing apostrophes except if the last letter before the apostrophe is 's'
def stripApo(word):
    strippedWord = ""
    for char in word:
        strippedWord = word.lstrip("'")
        if word.endswith("'") and not word.endswith("s'"):
            strippedWord = strippedWord.rstrip("'") 
    return strippedWord

# build FSA based on model description in modelFile
for line in open(modelFile, 'r').readlines():
    for char in line:
    # reads the line and builds individual words only from capital and lowercase letters + apostrophes
        if char.isalpha() or char == "'":
            cleanWord = cleanWord + char
        # if a character other than a letter or apostrophe is encountered, the word is converted to lowercase,
        # then stripped of leading or trailing apostrophes as I noted above
        else:
            cleanWord = cleanWord.lower()
            cleanWord = stripApo(cleanWord)
            if not cleanWord in dictW:
                dictW[cleanWord] = 1
            else:
                dictW[cleanWord] = dictW[cleanWord] + 1
            cleanWord = ""

# generates a list of just the frequencies
for token in dictW:
    if len(token) > 0:
        freqs.append(dictW[token])

# writes CSV file of sorted frequencies, plus rank numbers
outFile.write("Rank,Frequency\n")
for freqW in sorted(freqs, reverse = True):
    outFile.write(str(rank) + "," + str(freqW) + "\n")
    rank = rank + 1

outFile.close()
