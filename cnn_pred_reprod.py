# codes adapted/modified from https://github.com/dennybritz/cnn-text-classification-tf

import tensorflow as tf
from sklearn import metrics
import os
import pickle
from lib.rep_setting import ph_model_settings, gh_model_settings, vz_model_settings

settings = [ph_model_settings, gh_model_settings, vz_model_settings]


def evaluate():
    for setting in settings:
        print "\nEvaluating...{}\n".format(setting['lang'])
        # Evaluation
        # ==================================================
        avg_precision, avg_recall, avg_f1 = 0.0, 0.0, 0.0
        with open(setting['test_path'], 'r') as f:
            test_sets = pickle.load(f)
        for i, s in enumerate(setting['setting']):
            checkpoint_dir = os.path.join(setting['checkpoint_dir'], '{}/checkpoints/'.format(i))
            checkpoint_file = os.path.join(checkpoint_dir, "model-fold{}".format(i))
            graph = tf.Graph()
            with graph.as_default():
                session_conf = tf.ConfigProto(
                  allow_soft_placement=s['allow_soft_placement'],
                  log_device_placement=s['log_device_placement'])
                sess = tf.Session(config=session_conf)
                with sess.as_default():
                    # Load the saved meta graph and restore variables
                    saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
                    saver.restore(sess, checkpoint_file)

                    # Get the placeholders from the graph by name
                    input_x = graph.get_operation_by_name("input_x").outputs[0]
                    # input_y = graph.get_operation_by_name("input_y").outputs[0]
                    dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

                    # Tensors we want to evaluate
                    predictions = graph.get_operation_by_name("output/predictions").outputs[0]

                    x_test, y_test = test_sets[i]

                    # Collect the predictions here
                    predictions = sess.run(predictions, {input_x: x_test, dropout_keep_prob: 1.0})
                    precision, recall, f1, _ = metrics.precision_recall_fscore_support(y_true=y_test, y_pred=predictions, average='binary')
                    # print "fold {} - precision: {}, recall: {}, f1: {}".format(i, precision, recall, f1)
                    avg_precision += precision/5.0
                    avg_recall += recall/5.0
                    avg_f1 += f1/5.0
        print "dataset {} - average precision: {}, recall: {}, f1: {}".format(setting['lang'], avg_precision, avg_recall, avg_f1)


if __name__ == "__main__":
    evaluate()
