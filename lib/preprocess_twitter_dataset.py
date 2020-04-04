import click
from nltk.stem.snowball import EnglishStemmer, SpanishStemmer, stopwords
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
        if stopwordPath:
            self.stopwords = self.load_stopword(path=stopwordPath)
        else:
            self.stopwords = ('####_NONE_####')
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
            for line in reader:
                text = line['text'].decode('utf-8').lower()
                # text = http.sub('http_link', text)
                tokens = rule.findall(text)
                if self.stopwords:
                    tokens = self.removeStopwords(line=tokens)
                label = line[self.label]
                if len(tokens) > 0:
                    i = 0
                    while i < len(line):
                        yield tokens[i : i + self.max_sentence_length], label
                        i += self.max_sentence_length
                else:
                    continue


def process_election(lang, data_path, stopword_path, save_path):
    if lang == "English":
        stemmer = EnglishStemmer()
    elif lang == "Spanish":
        stemmer = SpanishStemmer()
    else:
        stemmer = None

    print "loading dataset"
    line_sentences = ProcessLineSentence(dataPath=data_path, label="election", stopwordPath=stopword_path, stemmer=stemmer)

    with open(save_path, 'w') as f:
        writer = csv.writer(f)
        for sentence, label in line_sentences:
            if label == "yes":
                l = [1]
            else:
                l = [0]
            row = [w.encode('utf-8') for w in sentence]
            writer.writerow(l+row)


@click.command()
@click.option("--lang", type=click.Choice(['English', 'Spanish', 'None']), help="language of the dataset")
@click.option("--data_path", type=click.Path(exists=True), help="path to the downloaded tweets dataset")
@click.option("--stopword_path", type=click.Path(exists=True), help="path to stopword file")
@click.option("--save_path", type=click.Path(), help="path to save tokenized data")
def run_election_process(lang, data_path, stopword_path, save_path):
    process_election(lang, data_path, stopword_path, save_path)


def process_violence(lang, data_path, stopword_path, save_path):
    if lang == "English":
        stemmer = EnglishStemmer()
    elif lang == "Spanish":
        stemmer = SpanishStemmer()
    else:
        stemmer = None

    print "loading dataset"
    line_sentences = ProcessLineSentence(dataPath=data_path, label="violence", stopwordPath=stopword_path, stemmer=stemmer)

    with open(save_path, 'w') as f:
        writer = csv.writer(f)
        for sentence, label in line_sentences:
            if label == "no":
                l = [0]
            elif label == "violence":
                l = [1]
            elif label == "malpractice":
                l = [2]
            else:
                raise(Exception("Wrong label: {}".format(label)))

            writer.writerow(l+sentence)


@click.command()
@click.option("--lang", type=click.Choice(['English', 'Spanish', 'None']), help="language of the dataset")
@click.option("--data_path", type=click.Path(exists=True), help="path to the downloaded tweets dataset")
@click.option("--stopword_path", type=click.Path(exists=True), help="path to stopword file")
@click.option("--save_path", type=click.Path(), help="path to save processed data")
def run_violence_process(lang, data_path, stopword_path, save_path):
    process_violence(lang, data_path, stopword_path, save_path)

if __name__ == "__main__":
    run_violence_process()
    # run_election_process()
