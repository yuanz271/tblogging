import os

import tensorflow as tf

__all__ = ["TBLogger"]


class TBLogger:
    def __init__(self, logdir, subfolder):
        self.subfolder = subfolder
        self.logdir = os.path.join(
            logdir, subfolder
        )
        self.writer = tf.summary.FileWriter(logdir=self.logdir)
        self.placeholders = {}
        self.summarize = None
        self.frozen = False

        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)

    def freeze(self):
        if self.frozen:
            return
        with self.graph.as_default():
            self.summarize = tf.summary.merge_all(key=self.subfolder)
        self.frozen = True

    def get_placeholders(self, category):
        placeholders = self.placeholders.get(category)
        if placeholders is None:
            placeholders = self.placeholders[category] = {}
        return placeholders

    def register_scalar(self, tag, category):
        with self.graph.as_default():
            with tf.variable_scope(category):
                placeholder = self.placeholders[tag] = tf.placeholder(
                    dtype=tf.float32, shape=(), name=tag
                )
            tf.summary.scalar(
                name=f"{category}/{tag}", tensor=placeholder, collections=[self.subfolder]
            )
        self.frozen = False

    def register_image(self, tag, category):
        with self.graph.as_default():
            with tf.variable_scope(category):
                placeholder = self.placeholders[tag] = tf.placeholder(
                    dtype=tf.uint8, shape=[1, None, None, 3], name=tag
                )
            tf.summary.image(
                name=f"{category}/{tag}", tensor=placeholder, collections=[self.subfolder]
            )
        self.frozen = False

    def log(self, step, feed_dict):
        self.freeze()
        feed_dict = {self.placeholders[k]: v for k, v in feed_dict.items()}
        summary = self.sess.run(self.summarize, feed_dict=feed_dict)
        self.writer.add_summary(summary, step)

    def close(self):
        self.writer.close()
        self.sess.close()
