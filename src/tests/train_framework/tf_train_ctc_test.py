#!/usr/bin/env python3
import unittest
import os
import logging
import tensorflow as tf

# custom modules
import train_framework.tf_train_ctc as tf_train


class TestTrain_ctc(unittest.TestCase):
    logging.basicConfig(level=logging.DEBUG)

    def setUp(self):
        '''
        Create the Tf_train_ctc instance
        '''
        self.tf_train_ctc = tf_train.Tf_train_ctc(
            config_file='neural_network.ini',
            debug=True,
            model_name=None)

    def tearDown(self):
        '''
        Close TF session if available
        '''
        if hasattr(self.tf_train_ctc, 'sess'):
            self.tf_train_ctc.sess.close()
            self.tf_train_ctc.writer.flush()

    def test_setup_tf_train_framework(self):
        '''
        Does the instance have the expected fields (i.e. type casting)
        '''
        tf_train_ctc = self.tf_train_ctc

        # make sure everything is loaded correctly
        self.assertEqual(isinstance(tf_train_ctc.epochs, int), True)
        self.assertEqual(isinstance(tf_train_ctc.network_type, str), True)
        self.assertEqual(isinstance(tf_train_ctc.n_input, int), True)
        self.assertEqual(isinstance(tf_train_ctc.n_context, int), True)
        self.assertEqual(isinstance(tf_train_ctc.model_dir, str), True)
        self.assertEqual(isinstance(tf_train_ctc.session_name, str), True)
        self.assertEqual(isinstance(tf_train_ctc.SAVE_MODEL_EPOCH_NUM, int), True)
        self.assertEqual(isinstance(tf_train_ctc.VALIDATION_EPOCH_NUM, int), True)
        self.assertEqual(isinstance(tf_train_ctc.CURR_VALIDATION_LER_DIFF, float), True)
        self.assertNotEqual(tf_train_ctc.beam_search_decoder, None)
        self.assertEqual(isinstance(tf_train_ctc.AVG_VALIDATION_LERS, list), True)
        self.assertEqual(isinstance(tf_train_ctc.shuffle_data_after_epoch, bool), True)
        self.assertEqual(isinstance(tf_train_ctc.min_dev_ler, float), True)

        # tests associated with data_sets object
        self.assertNotEqual(tf_train_ctc.data_sets, None)
        self.assertTrue(tf_train_ctc.sets == ['train', 'dev', 'test'])  # this will vary if changed in future
        self.assertEqual(isinstance(tf_train_ctc.n_examples_train, int), True)
        self.assertEqual(isinstance(tf_train_ctc.n_examples_dev, int), True)
        self.assertEqual(isinstance(tf_train_ctc.n_examples_test, int), True)
        self.assertEqual(isinstance(tf_train_ctc.batch_size, int), True)
        self.assertEqual(isinstance(tf_train_ctc.n_batches_per_epoch, int), True)

        # make sure folders are made
        self.assertTrue(os.path.exists(tf_train_ctc.SESSION_DIR))
        self.assertTrue(os.path.exists(tf_train_ctc.SUMMARY_DIR))

        # since model_name set as None, model_path should be None
        self.assertEqual(tf_train_ctc.model_path, None)

    def test_run_tf_train_gpu(self):
        '''
        Can a small model run on the GPU?
        '''
        tf_train_ctc = self.tf_train_ctc
        tf_train_ctc.run_model()

        # Verify objects of the train framework were created in the test training run
        self.assertNotEqual(tf_train_ctc.graph, None)
        self.assertNotEqual(tf_train_ctc.sess, None)
        self.assertNotEqual(tf_train_ctc.writer, None)
        self.assertNotEqual(tf_train_ctc.saver, None)
        self.assertNotEqual(tf_train_ctc.logits, None)

        # Verify some learning has been done
        self.assertTrue(tf_train_ctc.train_ler > 0)

        # make sure the input targets (i.e. .txt files) are not empty string
        self.assertTrue(tf_train_ctc.dense_labels is not '')

        # shutdown the running model
        self.tearDown()

    def test_verify_if_cpu_can_be_used(self):
        '''
        Can a small model run on the CPU?
        '''
        tf_train_ctc = self.tf_train_ctc
        tf_train_ctc.tf_device = '/cpu:0'

        tf_train_ctc.graph = tf.Graph()
        with tf_train_ctc.graph.as_default(), tf.device(tf_train_ctc.tf_device):
            with tf.device(tf_train_ctc.tf_device):
                tf_train_ctc.setup_network_and_graph()
                tf_train_ctc.load_placeholder_into_network()
                tf_train_ctc.setup_loss_function()
                tf_train_ctc.setup_optimizer()
                tf_train_ctc.setup_decoder()
                tf_train_ctc.setup_summary_statistics()
                tf_train_ctc.sess = tf.Session(
                    config=tf.ConfigProto(allow_soft_placement=True), graph=tf_train_ctc.graph)

                # initialize the summary writer
                tf_train_ctc.writer = tf.summary.FileWriter(
                    tf_train_ctc.SUMMARY_DIR, graph=tf_train_ctc.sess.graph)

                # Add ops to save and restore all the variables
                tf_train_ctc.saver = tf.train.Saver()

        # Verify objects of the train framework were created in the test training run
        self.assertNotEqual(tf_train_ctc.graph, None)
        self.assertNotEqual(tf_train_ctc.sess, None)
        self.assertNotEqual(tf_train_ctc.writer, None)
        self.assertNotEqual(tf_train_ctc.saver, None)

        # shutdown the running model
        self.tearDown()


if __name__ == '__main__':
    unittest.main()
