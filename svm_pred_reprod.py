from sklearn import metrics
from sklearn.externals import joblib
import os


def load_testset(lang):
    test_dir = os.path.join('test', '{}_svm_test'.format(lang))
    test_file = os.path.join(test_dir, '{}_test.joblib'.format(lang))
    test_sets = joblib.load(test_file)
    return test_sets


def load_svm_model(lang, fold):
    model_dir = os.path.join('logs/svm', '{}'.format(lang))
    model_file = os.path.join(model_dir, 'svm_model_fold{}.joblib'.format(fold))
    model = joblib.load(model_file)
    return model


def SVM_model_TFIDF(lang):
    test_data = load_testset(lang)
    predictions = []
    k = 5

    tp = []
    tr = []
    tf1 = []
    tbac = []
    for idx, test in enumerate(test_data):
        print "-------------- fold count...: {} ----------------".format(idx)
        x_test_vec = test[0]
        y_test = test[1]

        print "-- debug info --"
        print "example of training input: {}".format(x_test_vec[0])
        print "example of training labels: {}".format(y_test[:10])
        print "-- debug info end --\n"

        print 'loading LinearSVC model...'

        clf = load_svm_model(lang, fold=idx)

        tests = clf.predict(x_test_vec)
        predictions.append(tests.tolist())

        test_precision = metrics.precision_score(y_test, tests, average='binary')
        test_recall = metrics.recall_score(y_test, tests, average='binary')
        recall_two = metrics.recall_score(y_test, tests, average=None)
        test_f1 = metrics.f1_score(y_test, tests, average='binary')
        print "test: "
        print test_precision, test_recall, test_f1, sum(recall_two)/2
        print "end\n"

        tp.append(test_precision)
        tr.append(test_recall)
        tf1.append(test_f1)
        tbac.append(sum(recall_two)/2)

    print "---------------------- test ---------------"
    print "average precision_score: {}".format(sum(tp)/k)
    print "average recall_score: {}".format(sum(tr)/k)
    print "average f1_score: {}".format(sum(tf1)/k)
    print "average bac_score: {}".format(sum(tbac)/k)


def tfidf():
    print "Running SVM + tfidf"
    for lang in ['ph', 'gh', 'vz']:
        SVM_model_TFIDF(lang=lang)


if __name__ == "__main__":
    tfidf()
