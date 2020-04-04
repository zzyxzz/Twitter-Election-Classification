# -*- coding: utf-8 -*-
# codes adapted/modified from https://github.com/dennybritz/cnn-text-classification-tf

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import lib.data_helpers_trained as data_helpers
from lib.data_loader import get_WE_vectors, load_dataset
from model.cnn_model import TextCNN_Weighted
from lib.early_stop import EarlyStopping
from sklearn import metrics, cross_validation
import sys

# Parameters
# ==================================================
np.set_printoptions(threshold=sys.maxint)
script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))

parameters = ["dataset_path","vocab_path","vector_path","ms_path","oov_mode","window_size","embedding_dim",
              "filter_sizes","num_filters","dropout_keep_prob","l2_reg_lambda","lang","batch_size","num_epochs",
              "evaluate_every","checkpoint_every"]

# Dataset
tf.flags.DEFINE_string("dataset_path", "", "path to the dataset")
tf.flags.DEFINE_string("vocab_path", "", "path to the vocabulary")
tf.flags.DEFINE_string("vector_path", "", "path to embedding vector")
tf.flags.DEFINE_string("ms_path", "", "path to stats of embedding vector")

#OOV strategy
tf.flags.DEFINE_string("oov_mode", "", "OOV mode (zero, random, pure)")

# Model Hyperparameters
tf.flags.DEFINE_integer("window_size", 5, "window size of embeddings (default: 5)")
tf.flags.DEFINE_integer("embedding_dim", 500, "Dimensionality of character embedding (default: 500)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 200, "Number of filters per filter size (default: 200)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.01, "L2 regularizaion lambda (default: 0.01)")
tf.flags.DEFINE_string("lang", None, "dataset/language used (default: None)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 50, "Evaluate model on dev set after this many steps (default: 50)")
tf.flags.DEFINE_integer("checkpoint_every", 1000, "Save model after this many steps (default: 1000)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
print "\nParameters:"
for a in parameters:
    print "{}={}".format(a,FLAGS.__getattr__(a))

if not FLAGS.lang:
    raise Exception("Please set language/country!")

settings = "logs/cnn/{}_w{}_d{}".format(FLAGS.lang, FLAGS.window_size, FLAGS.embedding_dim)
script_dir = os.path.join(script_dir, settings)

if not os.path.exists(script_dir):
    os.makedirs(script_dir)


def run_weighted_CNN_model(dataset_path, vocab_path, vector_path, msPath, mode, trainable=True):
    # Data Preparatopn
    # ==================================================
    start_time = time.time()
    ts = str(int(time.time()))
    if mode == "local":
        ts += "_local"
    elif mode == "pure":
        ts += "_pure"
    elif mode == "zero":
        ts += "_zero"
    elif mode == "normal":
        ts += "_normal"
    elif mode == "random":
        ts += '_random'
    else:
        pass

    res_dir = os.path.join(script_dir, "res_{}_weighted".format(ts))

    shuffled_x, shuffled_y, shuffled_raw_y, vocabulary, vocabulary_inv, exist_idx = load_dataset(
        dataset_path=dataset_path, vocab_path=vocab_path)

    print "Vocabulary Size: {:d}".format(len(vocabulary))

    print "loading pre-trained vectors..."
    if vector_path:
        trained_vectors = get_WE_vectors(vector_path, msPath, exist_idx, vocabulary, mode)
        print "final word vectors size: ({}, {})".format(*trained_vectors.shape)
    else:
        trained_vectors = None
        print "use model embedding instead of pre-trained embedding"

    print "Start tensorflow..."
    results_dir = res_dir
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    x_train, x_test, y_train, y_test, raw_y_train, raw_y_test = \
        cross_validation.train_test_split(shuffled_x, shuffled_y, shuffled_raw_y, test_size=0.2, stratify=shuffled_raw_y)
    print "example of training input: {}".format(x_train[0])
    print "example of training label: {}".format(y_train[0])

    x_train, x_validation, y_train, y_validation = \
        cross_validation.train_test_split(x_train, y_train, test_size=0.2, stratify=raw_y_train)

    max_len = x_train.shape[1]
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN_Weighted(
                sequence_length=x_train.shape[1],
                num_classes=2,
                vocab_size=len(vocabulary),
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=map(int, FLAGS.filter_sizes.split(",")),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda,
                trainable=trainable)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-4)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step, name='train_op')

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.merge_summary(grad_summaries)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(script_dir, "runs", timestamp))
            print "Writing to {}\n".format(out_dir)

            # Summaries for loss and accuracy
            loss_summary = tf.scalar_summary("loss", cnn.loss)
            acc_summary = tf.scalar_summary("accuracy", cnn.accuracy)

            # Train Summaries
            train_summary_op = tf.merge_summary([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model-election")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.all_variables())

            # Initialize all variables
            sess.run(tf.initialize_all_variables())
            ''' override WE in CNN setup'''
            if trained_vectors:
                sess.run(cnn.W.assign(trained_vectors))

            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                r = 1.0
                labels = np.argmax(y_batch, axis=1)
                ratio_1 = 1.0 - len(labels[labels==1]) / float(len(labels))
                ratio_0 = 1.0 - ratio_1
                # print "weighted ratios ({}, {})".format(ratio_0, ratio_1)

                weighted_ratio = np.array([ratio_0, r * ratio_1]).reshape([1, 2])

                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: FLAGS.dropout_keep_prob,
                    cnn.weighted_ratio: weighted_ratio
                }
                _, step, summaries, loss, accuracy, predictions, ground = sess.run(
                    [train_op, global_step, train_summary_op,
                     cnn.loss, cnn.accuracy, cnn.predictions, cnn.ground],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()

                if step%10==0:
                    print "\nTraining:"
                    print "{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy)

                train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch, y_batch, test_set=False, writer=None):
                """
                Evaluates model on a dev set
                """
                r = 1.0
                labels = np.argmax(y_batch, axis=1)
                ratio_1 = 1.0 - len(labels[labels==1]) / float(len(labels)) ## ratio of violence instances
                ratio_0 = 1.0 - ratio_1
                # print "weighted ratios ({}, {})".format(ratio_0, ratio_1)
                weighted_ratio = np.array([ratio_0, r * ratio_1]).reshape([1, 2])
                # weighted_ratio =  np.ones([1, 2]

                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: 1.0,
                    cnn.weighted_ratio: weighted_ratio
                }

                step, summaries, loss, accuracy, predictions, ground, scores = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy, cnn.predictions, cnn.ground, cnn.scores],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()

                precision = metrics.precision_score(ground, predictions, average='binary')
                recall = metrics.recall_score(ground, predictions,  average='binary')
                f1_score = metrics.f1_score(ground, predictions, average='binary')

                precision_two = metrics.precision_score(ground, predictions, average=None)
                recall_two = metrics.recall_score(ground, predictions, average=None)
                f1_score_two = metrics.f1_score(ground, predictions, average=None)

                scores = {'precision': precision, 'recall': recall, 'f1score': f1_score,
                          'precision2': precision_two.tolist(), 'recall2': recall_two.tolist(),
                          'f1score2': f1_score_two.tolist(), 'loss': float(loss)}

                if test_set:
                    print "in test -- {}: loss {:g}, acc {:g}, f1_score {:g}".format(time_str, loss, accuracy, f1_score)
                else:
                    print "in validation -- {}: loss {:g}, acc {:g}, f1_score {:g}".format(time_str, loss, accuracy, f1_score)
                if writer:
                    writer.add_summary(summaries, step)
                return loss, predictions, ground, scores

            # Generate batches
            batches = data_helpers.batch_iter(
                zip(x_train, y_train), FLAGS.batch_size, FLAGS.num_epochs)

            # early_stop = EarlyStopping(threshold=8)
            # early_stop = None
            # Training loop. For each batch...
            best_val = 0.0
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print "validation at step: {}".format(current_step)
                    val_loss, _, _, val_results = dev_step(x_validation, y_validation, False, dev_summary_writer)
                    val_score = val_results['loss']

                    if val_score < best_val:
                        best_val = val_score
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        print "Saved model checkpoint to {}\n".format(path)

            test_loss, _, _, test_results = dev_step(x_test, y_test, True)
            print "test scores:\n{}".format(test_results)

    print "---------------all finished!--------------"
    print "--------------- {} minutes ----------------".format((time.time() - start_time) / 60.0)


if __name__ == "__main__":
    print "Runing CNN model..."
    run_weighted_CNN_model(dataset_path=FLAGS.dataset_path, vocab_path=FLAGS.vocab_path, vector_path=FLAGS.vector_path,
                  msPath=FLAGS.ms_path, mode=FLAGS.oov_mode)
