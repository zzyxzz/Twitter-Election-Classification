import numpy as np
from sklearn import metrics, cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
import click
import csv


def SVM_model_TFIDF(dataset_path, lang):
    decision_functions = []
    predictions = []

    def load_dataset_tfidf():
        data = []
        labels = []
        with open(dataset_path) as f:
            reader = csv.reader(f)
            for line in reader:
                label = int(line[0])
                if label == 2:
                    label = 0
                labels.append(label)
                data.append(' '.join(line[1:]))
        data = np.array(data)
        labels = np.array(labels)

        # Randomly shuffle data using a constant seed
        shuffle_indices = np.random.permutation(np.arange(len(labels)))
        print "shuffled index: {}".format(shuffle_indices[:10])

        shuffled_data = data[shuffle_indices]
        shuffled_labels = labels[shuffle_indices]

        print "ENTIRE DATASET --> positive: {} negative: {} total: {}".format(sum(labels), len(labels) - sum(labels), len(labels))
        return shuffled_data, shuffled_labels

    shuffled_x, shuffled_y = load_dataset_tfidf()

    x_train, x_test, y_train, y_test = \
        cross_validation.train_test_split(shuffled_x, shuffled_y, test_size=0.2, stratify=shuffled_y)

    x_train, x_validation, y_train, y_validation = \
        cross_validation.train_test_split(x_train, y_train, test_size=0.2, stratify=y_train)

    #vecterize text data into tfidf
    tfidf_vectorizer = TfidfVectorizer()
    x_train_vec = tfidf_vectorizer.fit_transform(x_train)
    x_validation_vec = tfidf_vectorizer.transform(x_validation)
    x_test_vec = tfidf_vectorizer.transform(x_test)

    print "-- debug info --"
    print "example of training input: {}".format(x_test_vec[0])
    print "example of training labels: {}".format(y_test[:10])
    print "-- debug info end --\n"

    print 'loading LinearSVC model...'
    if lang=="vz":
        c = 1
    elif lang == "ph":
        c = 1
    else:
        c = 1

    clf = svm.LinearSVC(C=c, class_weight='balanced')
    clf.fit(x_train_vec, y_train)

    validations = clf.predict(x_validation_vec)
    val_precision = metrics.precision_score(y_validation, validations, average='binary')
    val_recall = metrics.recall_score(y_validation, validations, average='binary')
    val_f1 = metrics.f1_score(y_validation, validations, average='binary')
    print "validation: "
    print val_precision, val_recall, val_f1
    print "\n"

    tests = clf.predict(x_test_vec)
    predictions.append(tests.tolist())

    tests_decision = clf.decision_function(x_test_vec)
    decision_functions.append(tests_decision.tolist())

    test_precision = metrics.precision_score(y_test, tests, average='binary')
    test_recall = metrics.recall_score(y_test, tests, average='binary')
    recall_two = metrics.recall_score(y_test, tests, average=None)
    test_f1 = metrics.f1_score(y_test, tests, average='binary')
    print "test: "
    print test_precision, test_recall, test_f1, sum(recall_two)/2
    print "end\n"

    test_accuracy = metrics.accuracy_score(y_test, tests)
    test_bac = sum(recall_two)/2

    results = {}
    results['precision'] = test_precision
    results['recall'] = test_recall
    results['f1'] = test_f1
    results['accuracy'] = test_accuracy
    results['recall2'] = recall_two.tolist()
    results['parac'] = c

    print "---------------------- validation ---------------"
    print "precision_score: {}".format(val_precision)
    print "recall_score: {}".format(val_recall)
    print "f1_score: {}".format(val_f1)

    print "---------------------- test ---------------"
    print "precision_score: {}".format(test_precision)
    print "recall_score: {}".format(test_recall)
    print "f1_score: {}".format(test_f1)
    print "bac_score: {}".format(test_bac)


@click.command()
@click.option("--dataset_path", type=click.Path(exists=True), help="path to dataset")
@click.option("--lang", type=click.STRING, help="language/country")
def tfidf(dataset_path, lang):
    print "Running SVM + tfidf"
    SVM_model_TFIDF(dataset_path=dataset_path, lang=lang)


if __name__ == "__main__":
    tfidf()
