# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 10:26:27 2021

@author: 19526
"""

import tensorflow as tf
import numpy as np
from model.attention import AdditiveAttention

class TextEncoder(object):
    def __init__(self, config, pretrained_word_embedding,keep_prob):
        self.config = config
        self.pretrained_word_embedding=pretrained_word_embedding
        self.keep_prob = keep_prob
        self.title_attention = AdditiveAttention(config.query_vector_dim,config.num_filters)

    def encoder(self,news):
        """
        Args:
            news:
                {
                    "text": batch_size * seq_length
                }
        Returns:
            (shape) batch_size, num_filters
        """
        with tf.variable_scope("text_encoder", reuse=tf.AUTO_REUSE) as scope:
            self.embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.num_filters])
            # batch_size,seq_length,num_filters
            self.embedding_inputs = tf.nn.embedding_lookup(self.embedding,news)
            embedding_inputs = tf.nn.dropout(self.embedding_inputs, rate=1-self.keep_prob)
            # batch_size,(seq_length-kernel_size+1),num_filters   ????????????????
            conv = tf.layers.conv1d(embedding_inputs, self.config.num_filters, self.config.kernel_size,padding='same', name='conv1')
            fc = tf.nn.relu(conv)
            # fc = self.multihead_self_attention.attention(embedding_inputs)
            fc = tf.nn.dropout(fc, rate=1-self.keep_prob)
            weighted_title_vector = self.title_attention.attention(fc)
            return weighted_title_vector

class ElementEncoder(object):
    def __init__(self, config,linear_output_dim):
        self.linear_output_dim=linear_output_dim
        self.config=config
    
    def encoder(self,element):
        self.embedding = tf.get_variable('element_embedding', [self.config.category_num, self.config.embedding_dim])
        return tf.nn.relu(tf.layers.dense(tf.nn.embedding_lookup(self.embedding,element),self.linear_output_dim))
    
class NewsEncoder(object):
    def __init__(self, config, pretrained_word_embedding,keep_prob):
        self.config = config
        self.pretrained_word_embedding=pretrained_word_embedding
        self.keep_prob = keep_prob
        self.final_attention = AdditiveAttention(config.query_vector_dim,config.num_filters)
        self.text_encoder=TextEncoder(config,pretrained_word_embedding,keep_prob)
        self.element_encoder=ElementEncoder(config,config.num_filters)
    """
        Args:
            "category": batch_size 
            "content": batch_size * num_words_title
        Returns:
            (shape) batch_size, num_filters
    """    
    def newsencoder(self,news,category):
        with tf.variable_scope("text_encoder", reuse=tf.AUTO_REUSE) as scope:
        # print(category)
            text_vectors = [self.text_encoder.encoder(news)]
            element_vectors= [self.element_encoder.encoder(category)]
            all_vectors = text_vectors + element_vectors
            # print(all_vectors)
            if len(all_vectors) == 1:
                final_news_vector=all_vectors[0]
            else:
                final_news_vector = self.final_attention.attention(tf.stack(all_vectors,axis=1))
        return final_news_vector

class UserEncoder(object):
    def __init__(self, config,keep_prob):
        self.config = config
        self.keep_prob = keep_prob
        self.additive_attention = AdditiveAttention(config.query_vector_dim,config.num_filters)

    def userencoder(self, user_vector):
        with tf.name_scope('user_encode'): 
            """
        Args:
            user_vector: batch_size, num_clicked_news_a_user, word_embedding_dim
        Returns:
            (shape) batch_size, word_embedding_dim
        """
            final_user_vector = self.additive_attention.attention(user_vector)
            return final_user_vector
        
class DotProductClickPredictor():
    def __init__(self,batch_size):
        self.batch_size=batch_size
        pass
    def predict(self, candidate_news_vector, user_vector):
        probability = tf.stack([tf.reduce_sum(tf.multiply(candidate_news_vector[i],user_vector[i]), axis=-1) for i in range(0,self.batch_size)],axis=0)
        probability = tf.nn.softmax(probability) #归一
        return probability
    
class Model():
    def __init__(self, config, pretrained_word_embedding=None):
        self.config = config
        self.input_click = tf.placeholder(tf.int32, [self.config.batch_size,self.config.click_len, self.config.num_words_title], name='input_click') #一个句子的长度
        self.input_click_category =  tf.placeholder(tf.int32, [self.config.batch_size,self.config.click_len], name='input_click_category') 
        self.input_candidate = tf.placeholder(tf.int32, [self.config.batch_size,self.config.candidate_len, self.config.num_words_title], name='input_candidate') #一个句子的长度
        self.input_candidate_category =  tf.placeholder(tf.int32, [self.config.batch_size,self.config.candidate_len], name='input_click_category') 
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')   #每个元素被保留的概率，防止过度拟
        self.news_encoder = NewsEncoder(config, pretrained_word_embedding,self.keep_prob)
        self.user_encoder = UserEncoder(config, self.keep_prob)
        self.click_predictor = DotProductClickPredictor(self.config.batch_size)
        self.main()
        

    def main(self):
        with tf.name_scope("nrml"):
            candidate_tp=tf.transpose(self.input_candidate,perm=[1,0,2])
            candidate_cat_tp=tf.transpose(self.input_candidate_category,perm=[1,0])
            # batch_size,candidate_len,num_filters
            candidate_news_vector = tf.stack([self.news_encoder.newsencoder(candidate_tp[x],candidate_cat_tp[x]) for x in range(0,self.config.candidate_len)],axis=1)
            click_tp=tf.transpose(self.input_click,perm=[1,0,2])
            click_cat_tp=tf.transpose(self.input_click_category,perm=[1,0])
            clicked_news_vector = tf.stack([self.news_encoder.newsencoder(click_tp[x],click_cat_tp[x]) for x in range(0,self.config.click_len)],axis=1)
            # batch_size, num_filters
            user_vector = self.user_encoder.userencoder(clicked_news_vector)
            # batch_size, candidate_num
            self.click_probability = self.click_predictor.predict(candidate_news_vector,user_vector)
        with tf.name_scope("optimize"):
            self.real_score=self.click_probability[:,0]
            # print(self.real_score.shape)
            self.input_real = tf.constant(1.0,dtype=tf.float32,shape=[self.config.batch_size],name='real') #,dtype=tf.float32
            # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.real_score, labels=self.input_real)
            self.loss = -tf.reduce_mean(self.real_score) #cross_entropy#
            # self.loss=tf.reduce_mean(cross_entropy)
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)
    
        with tf.name_scope("accuracy"):
            correct_pred = tf.equal(tf.argmax(self.click_probability,1),0)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
        return self.click_probability 
        