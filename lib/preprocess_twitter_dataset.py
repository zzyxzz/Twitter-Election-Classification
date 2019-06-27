import click
from nltk.stem.snowball import EnglishStemmer,SpanishStemmer
import csv
import os
import regex as re
from unidecode import unidecode

# rule = re.compile(r"[@#\p{Alpha}]\w{2,}", flags=re.UNICODE)
rule = re.compile(r"[@#\p{Alpha}][\w]+|[A-Z]{2,}|\d+", flags=re.UNICODE)
http = re.compile(r"https?[\w:./]+")

class ProcessLineSentence(object):
    """
    load EV dataset with csv format and preprcoess the tweet
    Modified from gensim.models.word2vec.LineSentence
    """

    def __init__(self, dataPath, label, stopwordPath, max_sentence_length=10000, limit=None, stemmer=None):
        self.dataPath = os.path.expanduser(dataPath)
        self.max_sentence_length = max_sentence_length
        self.limit = limit
        self.stopwords = self.load_stopword(path=stopwordPath)
        self.stemmer = stemmer
        self.label = label

    def load_stopword(self, path):
        stopwords = {}
        path=os.path.expanduser(path)
        with open(path, 'r') as f:
            for line in f:
                word = line.strip()
                stopwords[word] = ""
        return stopwords

    def isStopword(self, word):
        if word in self.stopwords:
            return True
        else:
            return False

    def removeStopwords(self, line):
        cleaned_line = []
        for w in line:
            if self.isStopword(w):
                continue
            else:
                if self.stemmer:
                    cleaned_line.append(unidecode(self.stemmer.stem(w)))
                else:
                    cleaned_line.append(unidecode(w))
        return cleaned_line

    def __iter__(self):
        """Iterate through the lines in the dataset."""
        with open(self.dataPath) as fin:
            reader = csv.DictReader(fin)
            _c = 0
            for line in reader:
                text = line['text'].decode('utf-8').lower()
                text = http.sub('http_link', text)
                tokens = rule.findall(text)
                tokens = self.removeStopwords(line=tokens)
                _c+=1
                if _c<5:
                    print(text)
                    print(tokens)
                label = line[self.label]
                if len(tokens) > 0:
                    i = 0
                    while i < len(line):
                        yield tokens[i : i + self.max_sentence_length], label
                        i += self.max_sentence_length
                else:
                    continue

# @click.option("--n_cluster", type=int, help="number of clusters")
@click.command()
@click.option("--lang", type=str, help="language of dataset")
@click.option("--data_path", type=str, help="path to dataset")
@click.option("--stopword_path", type=str, help="path to stopword list")
@click.option("--save_path", type=str, help="path to save tokens")
def runModel(lang, data_path, stopword_path, save_path):
    if lang == "English":
        stemmer = EnglishStemmer()
    else:
        stemmer = SpanishStemmer()
    # stemmer=None

    print("loading dataset")
    line_sentences = ProcessLineSentence(dataPath=data_path, label="election", stopwordPath=stopword_path, stemmer=stemmer)

    with open(save_path, 'w') as f:
        writer = csv.writer(f)
        for sentence, label in line_sentences:
            print(label,sentence)
            if label == "yes":
                l = [1]
            else:
                l = [0]
            print(l+sentence)
            writer.writerow(l+sentence)


@click.command()
@click.option("--lang", type=str, help="language of dataset")
@click.option("--data_path", type=str, help="path to dataset")
@click.option("--stopword_path", type=str, help="path to stopword list")
@click.option("--save_path", type=str, help="path to save tokens")
def runViolenceModel(lang, data_path, stopword_path, save_path):
    if lang == "English":
        stemmer = EnglishStemmer()
    else:
        stemmer = SpanishStemmer()
    print("loading dataset")
    # stemmer=None
    line_sentences = ProcessLineSentence(dataPath=data_path, label="violence", stopwordPath=stopword_path, stemmer=stemmer)

    with open(save_path, 'w') as f:
        writer = csv.writer(f)
        for sentence, label in line_sentences:
            print(label,sentence)
            if label == "no":
                l = [0]
            elif label == "violence":
                l = [1]
            elif label == "malpractice":
                l = [2]
            else:
                raise(Exception("Wrong label: {}".format(label)))
            print(l+sentence)
            writer.writerow(l+sentence)

if __name__ == "__main__":
    runViolenceModel()
    # runModel()
