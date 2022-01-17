# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 15:41:09 2020
"""
import tensorflow as tf
import numpy as np
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '/gpu:0'

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
        self.input_click_entity = tf.placeholder(tf.int32, [self.config.batch_size,self.config.click_len, self.config.num_words_title], name='input_click_entity')
        self.input_candidate = tf.placeholder(tf.int32, [self.config.batch_size,self.config.candidate_len, self.config.num_words_title], name='input_candidate') #一个句子的长度
        self.input_candidate_entity = tf.placeholder(tf.int32, [self.config.batch_size,self.config.candidate_len, self.config.num_words_title], name='input_candidate_entity') #一个句子的长度
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')   #每个元素被保留的概率，防止过度拟
        self.news_encoder = NewsEncoder(config, pretrained_word_embedding,self.keep_prob)
        self.user_encoder = UserEncoder(config,self.keep_prob)
        self.click_predictor = DotProductClickPredictor(self.config.batch_size)
        self.main()
        

    def main(self):
        candidate_tp=tf.transpose(self.input_candidate,perm=[1,0,2])
        candidate_entity_tp=tf.transpose(self.input_candidate_entity,perm=[1,0,2])
        # batch_size,candidate_len,num_filters
        candidate_news_vector = tf.stack([self.news_encoder.newsencoder(candidate_tp[x],candidate_entity_tp[x]) for x in range(0,self.config.candidate_len)],axis=1)        
        click_tp=tf.transpose(self.input_click,perm=[1,0,2])
        click_entity_tp=tf.transpose(self.input_click_entity,perm=[1,0,2])
        clicked_news_vector = tf.stack([self.news_encoder.newsencoder(click_tp[x],click_entity_tp[x]) for x in range(0,self.config.click_len)],axis=1)
        # batch_size, num_filters
        user_vector = self.user_encoder.userencoder(clicked_news_vector)
        # batch_size, candidate_num
        self.click_probability = self.click_predictor.predict(candidate_news_vector,user_vector)
        with tf.name_scope("optimize"):
            self.real_score=self.click_probability[:,0:self.config.real_num]
            # print(self.real_score.shape)
            self.input_real = tf.constant(1.0,dtype=tf.float32,shape=[self.config.batch_size],name='real') #,dtype=tf.float32
            # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.real_score, labels=self.input_real)
            self.loss = -tf.reduce_mean(tf.reduce_sum(self.real_score,axis=1)) #cross_entropy#
            # self.loss=tf.reduce_mean(cross_entropy)
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)
        with tf.name_scope("accuracy"):
            correct_pred = tf.equal(tf.argmax(self.click_probability,1),0)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        return self.click_probability 
    
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 20:15:19 2020

@author: 1952640
"""

from model.attention import AdditiveAttention
class NewsEncoder(object):
    def __init__(self, config, pretrained_word_embedding,keep_prob):
        self.config = config
        self.pretrained_word_embedding=pretrained_word_embedding
        self.keep_prob = keep_prob
        self.title_attention = AdditiveAttention(config.query_vector_dim,config.num_filters)
        #self.embedding = tf.Variable( tf.random_normal(shape=[self.config.vocab_size, self.config.num_filters],mean=0,stddev=1),name='embedding',)
        self.multihead_self_attention = MultiHeadSelfAttention(config.num_filters, config.num_attention_heads)
        
    def newsencoder(self,news,entities):
        """
        Args:
            news:
                {
                    "text": batch_size * seq_length
                }
        Returns:
            (shape) batch_size, num_filters
        """
        #with tf.device('/gpu:0'):
        with tf.variable_scope("news_encoder", reuse=tf.AUTO_REUSE) as scope:
            self.embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.num_filters])
            self.entity_embedding = tf.get_variable('entity_embedding',[self.config.entity_size,self.config.num_filters])
            # batch_size,seq_length,num_filters
            self.embedding_words = tf.nn.embedding_lookup(self.embedding,news)
            self.embedding_entities = tf.nn.embedding_lookup(self.entity_embedding,entities)
            embedding_inputs = tf.concat([self.embedding_words,self.embedding_entities],axis=-1)
            embedding_inputs = tf.nn.dropout(embedding_inputs, rate=1-self.keep_prob)
            
            # batch_size,(seq_length-kernel_size+1),num_filters*2   ????????????????
            conv = tf.layers.conv1d(embedding_inputs, self.config.num_filters*2, self.config.kernel_size,padding='same', name='conv1')   
                
            # print(conv.shape)
            fc = tf.nn.relu(conv) 
            
            # fc = self.multihead_self_attention.attention(embedding_inputs)
            fc = tf.nn.dropout(fc, rate=1-self.keep_prob)
            weighted_title_vector = self.title_attention.attention(fc)
            # weighted_title_vector = tf.reduce_max(fc, reduction_indices=[1])
            return weighted_title_vector
            
        

# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 16:05:43 2020

@author: 1952640
"""
from model.attention import MultiHeadSelfAttention

class UserEncoder(object):
    def __init__(self, config,keep_prob):
        self.config = config
        self.keep_prob = keep_prob
        self.multihead_self_attention = MultiHeadSelfAttention(config.num_filters*2, config.num_attention_heads)
        self.additive_attention = AdditiveAttention(config.query_vector_dim,config.num_filters)

    def userencoder(self, user_vector):
        #with tf.device('/gpu:0'):
        with tf.name_scope('user_encode'): 
            """
        Args:
            user_vector: batch_size, num_clicked_news_a_user, word_embedding_dim
        Returns:
            (shape) batch_size, word_embedding_dim
            
        def lstm_cell():
            return tf.contrib.rnn.BasicLSTMCell(self.config.num_filters, state_is_tuple=True)
        
        def gru_cell():
            return tf.contrib.rnn.GRUCell(self.config.num_filters)
        
        def dropout():
            if (self.config.model == 'lstm'):
                cell = lstm_cell()
            else:
                cell = gru_cell()
            return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
        with tf.name_scope("rnn"):
            cells = [dropout() for _ in range(self.config.kernel_size)]
            rnn_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)        
            user_vector, _ = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=user_vector, dtype=tf.float32)   
            """         
            # batch_size, num_clicked_news_a_user, word_embedding_dim
            # multihead_user_vector = user_vector
            # multihead_user_vector = user_vector
            # multihead_user_vector = self.multihead_self_attention.attention(user_vector)
            # batch_size, word_embedding_dim
            final_user_vector = self.additive_attention.attention(user_vector)#multihead_user_vector)
            #print(final_user_vector.shape)
            #final_user_vector = tf.reduce_max(user_vector, reduction_indices=[1])
            return final_user_vector