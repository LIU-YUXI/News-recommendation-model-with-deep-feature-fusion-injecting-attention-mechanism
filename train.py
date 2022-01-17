# -*- coding: utf-8 -*-

"""
Created on Wed Oct 21 22:49:38 2020

@author: 1952640
"""

import os
import sys
import json
import time
import math
from datetime import timedelta

import numpy as np
import tensorflow as tf
import pandas as pd
#import tensorflow.compat.v1 as tf
#tf.disable_eager_execution()
from sklearn import metrics

from model.__init__ import Model
from model.naml import Model as NAML
from model.dkn import Model as DKN
from model.config import CHConfig,ENConfig,DKNConfig
from dataset import read_vocab, read_category, batch_iter, process_file, build_vocab,test_process_file,test_batch_iter,online_process,open_file
import dataset_en as de
import dataset_en_naml as dem
import dataset_en_dkn as dek

base_dir = '../data'
train_dir = os.path.join(base_dir, 'train.txt')
test_dir = os.path.join(base_dir, 'test.txt')
val_dir = os.path.join(base_dir, 'val.txt')
vocab_dir ='../data'
vocab_dir = os.path.join(vocab_dir, 'vocab.txt')

save_dir = './checkpoints/en+'
save_path = os.path.join(save_dir, 'best_validation')  # æä½³éªè¯ç»æä¿å­è·¯

def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def feed_data(click, candidate,real, keep_prob):
    feed_dict = {
        model.input_click: click,
        model.input_candidate: candidate,
        #model.real_len: real,
        model.keep_prob: keep_prob
    }
    return feed_dict
    
def feed_data_naml(click,click_cat, candidate,candidate_cat,real, keep_prob):
    feed_dict = {
        model.input_click: click,
        model.input_click_category:click_cat,
        model.input_candidate: candidate,
        model.input_candidate_category:candidate_cat,
        #model.real_len: real,
        model.keep_prob: keep_prob
    }
    return feed_dict

def feed_data_dkn(click,click_entity, candidate,candidate_entity,real, keep_prob):
    feed_dict = {
        model.input_click: click,
        model.input_click_entity:click_entity,
        model.input_candidate: candidate,
        model.input_candidate_entity:candidate_entity,
        #model.real_len: real,
        model.keep_prob: keep_prob
    }
    return feed_dict

def evaluate(sess, x_, y_):
    batch_eval = batch_iter(x_, y_,batch_size=config.batch_size,max_length=config.num_words_title,candidate_num=config.candidate_len,click_num=config.click_len,real_num=config.real_num)
    total_loss = 0.0
    total_acc = 0.0
    total_mrr = 0.0
    count=0
    for click, candidate,real in batch_eval:
        count+=1
        #print(candidate.shape)
        feed_dict = feed_data(click, candidate,real, 1.0)
        loss, click_prob = sess.run([model.loss,model.click_probability], feed_dict=feed_dict)
        mrr=MRR(click_prob)
        acc=ACC(click_prob)
        total_loss += loss
        total_acc += acc 
        total_mrr += mrr
        if(count>80):
            break

    return total_loss / count, total_acc / count, total_mrr/ count


def evaluate_en(sess,users,news_content,content):
    batch_eval = de.batch_iter(users,news_content,content,
                                    batch_size=config.batch_size,max_length=config.num_words_title,candidate_size=config.candidate_len,
                                    click_size=config.click_len,real_num=config.real_num)
    total_loss = 0.0
    total_acc = 0.0 
    total_mrr = 0.0 
    count=0
    for click, candidate,real in batch_eval:
        count+=1
        #print(candidate.shape)
        feed_dict = feed_data(click, candidate,real, 1.0)
        loss, click_prob = sess.run([model.loss,model.click_probability], feed_dict=feed_dict)
        mrr=MRR(click_prob)
        acc=ACC(click_prob)
        total_loss += loss
        total_acc += acc 
        total_mrr += mrr
        if(count>80):
            break

    return total_loss / count, total_acc / count, total_mrr/ count

def evaluate_en_naml(sess,users,news_content,content,news_category,category):
    batch_eval = dem.batch_iter(users,news_content,content,news_category,category,
                                    batch_size=config.batch_size,max_length=config.num_words_title,candidate_size=config.candidate_len,
                                    click_size=config.click_len,real_num=config.real_num)
    total_loss = 0.0
    total_acc = 0.0 
    total_mrr = 0.0 
    count=0
    for click,click_cat, candidate,candidate_cat,real in batch_eval:
        count+=1
        #print(candidate.shape)
        feed_dict = feed_data_naml(click,click_cat,candidate,candidate_cat,real, 1.0)
        loss, click_prob = sess.run([model.loss,model.click_probability], feed_dict=feed_dict)
        mrr=MRR(click_prob)
        acc=ACC(click_prob)
        total_loss += loss
        total_acc += acc 
        total_mrr += mrr
        if(count>80):
            break

    return total_loss / count, total_acc / count, total_mrr/ count
    
def evaluate_en_dkn(sess,users,news_content,content,news_entity,entity):
    batch_eval = dek.batch_iter(users,news_content,content,news_entity,entity,
                                    batch_size=config.batch_size,max_length=config.num_words_title,candidate_size=config.candidate_len,
                                    click_size=config.click_len,real_num=config.real_num)
    total_loss = 0.0
    total_acc = 0.0 
    total_mrr = 0.0 
    count=0
    for click,click_entity, candidate,candidate_entity,real in batch_eval:
        count+=1
        #print(candidate.shape)
        feed_dict = feed_data_dkn(click,click_entity,candidate,candidate_entity,real, 1.0)
        loss, click_prob = sess.run([model.loss,model.click_probability], feed_dict=feed_dict)
        mrr=MRR(click_prob)
        acc=ACC(click_prob)
        total_loss += loss
        total_acc += acc 
        total_mrr += mrr
        if(count>80):
            break

    return total_loss / count, total_acc / count, total_mrr/ count

def train():
    print("Configuring TensorBoard and Saver...")
    tensorboard_dir = './tensorboard/ch'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy", model.acc)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)

    saver = tf.train.Saver()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("Loading training and validation data...")
    start_time = time.time()
    news_train, users_train = process_file(train_dir, word_to_id, cat_to_id, config.seq_length)
    news_val, users_val = process_file(val_dir, word_to_id, cat_to_id, config.seq_length)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
    
    configsess= tf.ConfigProto(allow_soft_placement=True)
    configsess.gpu_options.allow_growth = True
    session = tf.Session() # config=configsess)
    session.run(tf.global_variables_initializer())
    writer.add_graph(session.graph)

    print('Training and evaluating...')
    start_time = time.time()
    total_batch = 0  
    best_acc_val = 0.0 
    last_improved = 0  
    require_improvement = 200  
    
    flag = False
    for epoch in range(config.num_epochs):
        print('Epoch:', epoch + 1)
        batch_train = batch_iter(news_train, users_train, batch_size=config.batch_size,max_length=config.num_words_title,candidate_num=config.candidate_len,click_num=config.click_len,real_num=config.real_num)
        acc=0
        loss=0
        mrr=0
        for click, candidate,real in batch_train:
            feed_dict = feed_data(click, candidate, real, config.dropout_keep_prob)
            
            #print(x_batch.shape[0],x_batch.shape[1])
            if total_batch % config.save_per_batch == 0 :
                s = session.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(s, total_batch)

            if ((total_batch % config.print_per_batch == 0)):#and (total_batch!=0)):
                feed_dict[model.keep_prob] = 1.0
                #loss_train, acc_train = session.run([model.loss, model.acc], feed_dict=feed_dict)
                acc_train=acc/config.print_per_batch
                loss_train=loss/config.print_per_batch
                mrr_train=mrr/config.print_per_batch
                acc=0
                loss=0
                mrr=0
                loss_val, acc_val,mrr_val = evaluate(session, news_val, users_val)# todo
                
                if acc_val > best_acc_val:
                    best_acc_val = acc_val
                    last_improved = total_batch
                    saver.save(sess=session, save_path=save_path)
                    improved_str = '*'
                else:
                    improved_str = ''

                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6}, Train Loss: {1:>6.4}, Train nDCG: {2:>7.2%},Train Mrr: {3:6.4}' \
                      + ' Val Loss: {4:>6.4}, Val nDCG: {5:>7.2%}, Val Mrr: {6:6.4} Time: {7} {8}'
                print(msg.format(total_batch, loss_train, acc_train,mrr_train,loss_val, acc_val,mrr_val,time_dif, improved_str))
            
            feed_dict[model.keep_prob] = config.dropout_keep_prob
            _loss,click_prob,optim=session.run([model.loss,model.click_probability,model.optim], feed_dict=feed_dict)
            _mrr=MRR(click_prob)
            _acc=ACC(click_prob)
            loss+=_loss
            acc+=_acc
            mrr+=_mrr
            total_batch += 1
            if total_batch - last_improved > require_improvement:
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag: 
            break

def test():
    print("Loading test data...")
    start_time = time.time()
    news_test, user_test,contents = test_process_file(test_dir, word_to_id, cat_to_id, config.seq_length)

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)  
    '''
    print('Testing...')
    loss_test, acc_test = evaluate(session, x_test, y_test)
    msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
    print(msg.format(loss_test, acc_test))
    '''
    
    batch_test = test_batch_iter(news_test, user_test, batch_size=config.batch_size,max_length=config.num_words_title,candidate_num=config.candidate_len)
    count=0

    for click, candidate,real,nolist in batch_test:
        feed_dict = feed_data(click, candidate, real, 1.0)
        click_predict=session.run(model.click_probability, feed_dict=feed_dict)
        for i in range(1):
            # click 2 real 1 candidate 2
            print('\n user : ',user_test[nolist[i][2]])
            print(' click : ')
            for news in contents[nolist[i][0]:nolist[i][1]]:
                news_end = (30 if(len(news)>=30) else len(news))
                print("".join('%s'%news[k] for k in range(0,news_end)))
            print(' candidate sort: ')
            print('score : %.2f' % (click_predict[i][0]*100))
            news=contents[nolist[i][2]]
            news_end = (30 if(len(news)>=30) else len(news))
            print('content : ',"".join('%s' %news[k] for k in range(0,news_end)))
            for j in range(0,config.candidate_len-1) :
                print('score : %.2f' % (click_predict[i][j+1]*100))
                news=contents[nolist[i][3]+j]
                news_end = (30 if(len(news)>=30) else len(news))
                print('content : ',"".join('%s' %news[k] for k in range(0,news_end)))
                
        break
    
    # click_predict = np.array(np.expand_dims(click_predict,1))
    # click_expect = np.zeros(shape=(count,1), dtype=np.int32)
    # print("Precision, Recall and F1-Score...")
    # print(metrics.classification_report(click_expect, click_predict, target_names=['True']))

    # print("Confusion Matrix...")
    # cm = metrics.confusion_matrix(click_expect, click_predict)
    # print(cm)

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

def grade(data):
    NEWS=json.loads(data)
    clicked,candidate,real=online_process(NEWS['history'],NEWS['candidate'],word_to_id,config.num_words_title)#(NEWS.history,NEWS.candidate)
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)  
    feed_dict = feed_data([clicked], [candidate], real, 1.0)
    click_predict=session.run(model.click_probability, feed_dict=feed_dict)
    result = json.dumps(click_predict[0].tolist())
    print(result)
    sys.stdout.flush()
    
def train_en():
    print("Configuring TensorBoard and Saver...")
    tensorboard_dir = './tensorboard/en+'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy", model.acc)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)

    saver = tf.train.Saver()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("Loading training and validation data...")
    start_time = time.time()
    users=pd.read_csv("../data/endata/train.tsv")
    test=pd.read_csv("../data/endata/test.tsv")
    news=pd.read_csv("../data/endata/news_parsed.tsv",sep='\t',index_col=False)
    news_content,content=de.news_process(news)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
    
    configsess= tf.ConfigProto(allow_soft_placement=True)
    configsess.gpu_options.allow_growth = True
    session = tf.Session()# config=configsess)
    session.run(tf.global_variables_initializer())
    writer.add_graph(session.graph)

    print('Training and evaluating...')
    start_time = time.time()
    total_batch = 0  
    best_acc_val = 0.0 
    last_improved = 0  
    require_improvement = 200  
    
    flag = False
    for epoch in range(config.num_epochs):
        print('Epoch:', epoch + 1)
        batch_train = de.batch_iter(users,news_content,content,
                                    batch_size=config.batch_size,max_length=config.num_words_title,
                                    candidate_size=config.candidate_len,click_size=config.click_len,real_num=config.real_num)
        #batch_train = batch_iter(news_train, users_train, batch_size=config.batch_size,max_length=config.num_words_title,candidate_num=config.candidate_len)
        acc=0
        loss=0
        mrr=0
        for click, candidate,real in batch_train:
            feed_dict = feed_data(click, candidate, real, config.dropout_keep_prob)
            
            #print(x_batch.shape[0],x_batch.shape[1])
            if total_batch % config.save_per_batch == 0 :
                s = session.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(s, total_batch)

            if ((total_batch % config.print_per_batch == 0)):#and (total_batch!=0)):
                feed_dict[model.keep_prob] = 1.0
                #loss_train, acc_train = session.run([model.loss, model.acc], feed_dict=feed_dict)
                acc_train=acc/config.print_per_batch
                loss_train=loss/config.print_per_batch
                mrr_train=mrr/config.print_per_batch
                acc=0
                loss=0
                mrr=0
                loss_val, acc_val,mrr_val= evaluate_en(session,test,news_content,content)# todo
                
                if acc_val > best_acc_val:
                    best_acc_val = acc_val
                    last_improved = total_batch
                    saver.save(sess=session, save_path=save_path)
                    improved_str = '*'
                else:
                    improved_str = ''

                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6}, Train Loss: {1:>6.4}, Train nDCG: {2:>7.2%},Train Mrr: {3:6.4}' \
                      + ' Val Loss: {4:>6.4}, Val nDCG: {5:>7.2%}, Val Mrr: {6:6.4} Time: {7} {8}'
                print(msg.format(total_batch, loss_train, acc_train,mrr_train,loss_val, acc_val,mrr_val,time_dif, improved_str))
            
            feed_dict[model.keep_prob] = config.dropout_keep_prob
            #res_train = session.run(model.news_encoder.title_attention.attention_query_vector,feed_dict=feed_dict)
            #print(feed_dict)
            #print(res_train)
            #print(feed_dict)
            #session.run(model.optim, feed_dict=feed_dict)  # è¿è¡ä¼å
            _loss,click_prob,optim=session.run([model.loss,model.click_probability,model.optim], feed_dict=feed_dict)
            _mrr=MRR(click_prob)
            _acc=ACC(click_prob)
            loss+=_loss
            acc+=_acc
            mrr+=_mrr
            total_batch += 1
            '''
            if total_batch - last_improved > require_improvement:
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
            '''
            if ((total_batch % (10*config.print_per_batch) == 0)):
                break
        if flag: 
            break    
    
def train_en_dkn():
    print("Configuring TensorBoard and Saver...")
    tensorboard_dir = './tensorboard/endkn'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy", model.acc)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)

    saver = tf.train.Saver()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("Loading training and validation data...")
    start_time = time.time()
    users=pd.read_csv("../data/endata/train.tsv")
    test=pd.read_csv("../data/endata/test.tsv")
    news=pd.read_csv("../data/endata/news_parsed.tsv",sep='\t',index_col=False)
    news_content,content,news_entity,entity=dek.news_process(news)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
    
    configsess= tf.ConfigProto(allow_soft_placement=True)
    configsess.gpu_options.allow_growth = True
    session = tf.Session()# config=configsess)
    session.run(tf.global_variables_initializer())
    writer.add_graph(session.graph)

    print('Training and evaluating...')
    start_time = time.time()
    total_batch = 0  
    best_acc_val = 0.0 
    last_improved = 0  
    require_improvement = 200  
    
    flag = False
    for epoch in range(config.num_epochs):
        print('Epoch:', epoch + 1)
        batch_train = dek.batch_iter(users,news_content,content,news_entity,entity,
                                    batch_size=config.batch_size,max_length=config.num_words_title,
                                    candidate_size=config.candidate_len,click_size=config.click_len,real_num=config.real_num)
        #batch_train = batch_iter(news_train, users_train, batch_size=config.batch_size,max_length=config.num_words_title,candidate_num=config.candidate_len)
        acc=0
        loss=0
        mrr=0
        for click,click_entity, candidate,candidate_entity,real in batch_train:
            feed_dict = feed_data_dkn(click,click_entity, candidate,candidate_entity, real, config.dropout_keep_prob)
            
            #print(x_batch.shape[0],x_batch.shape[1])
            if total_batch % config.save_per_batch == 0 :
                s = session.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(s, total_batch)

            if ((total_batch % config.print_per_batch == 0)):#and (total_batch!=0)):
                feed_dict[model.keep_prob] = 1.0
                #loss_train, acc_train = session.run([model.loss, model.acc], feed_dict=feed_dict)
                acc_train=acc/config.print_per_batch
                loss_train=loss/config.print_per_batch
                mrr_train=mrr/config.print_per_batch
                acc=0
                loss=0
                mrr=0
                loss_val, acc_val,mrr_val= evaluate_en_dkn(session,test,news_content,content,news_entity,entity)# todo
                
                if acc_val > best_acc_val:
                    best_acc_val = acc_val
                    last_improved = total_batch
                    saver.save(sess=session, save_path=save_path)
                    improved_str = '*'
                else:
                    improved_str = ''

                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6}, Train Loss: {1:>6.4}, Train nDCG: {2:>7.2%},Train Mrr: {3:6.4}' \
                      + ' Val Loss: {4:>6.4}, Val nDCG: {5:>7.2%}, Val Mrr: {6:6.4} Time: {7} {8}'
                print(msg.format(total_batch, loss_train, acc_train,mrr_train,loss_val, acc_val,mrr_val,time_dif, improved_str))
            
            feed_dict[model.keep_prob] = config.dropout_keep_prob
            #res_train = session.run(model.news_encoder.title_attention.attention_query_vector,feed_dict=feed_dict)
            #print(feed_dict)
            #print(res_train)
            #print(feed_dict)
            #session.run(model.optim, feed_dict=feed_dict)  # è¿è¡ä¼å
            _loss,click_prob,optim=session.run([model.loss,model.click_probability,model.optim], feed_dict=feed_dict)
            _mrr=MRR(click_prob)
            _acc=ACC(click_prob)
            loss+=_loss
            acc+=_acc
            mrr+=_mrr
            total_batch += 1
            '''
            if total_batch - last_improved > require_improvement:
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
            '''
            if ((total_batch % (10*config.print_per_batch) == 0)):
                break
        if flag: 
            break    
    
def train_en_naml():
    print("Configuring TensorBoard and Saver...")
    tensorboard_dir = './tensorboard/ENNAML'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy", model.acc)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)

    saver = tf.train.Saver()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("Loading training and validation data...")
    start_time = time.time()
    users=pd.read_csv("../data/endata/train.tsv")
    test=pd.read_csv("../data/endata/test.tsv")
    news=pd.read_csv("../data/endata/news_parsed.tsv",sep='\t',index_col=False)
    news_content,content,news_category,category=dem.news_process(news)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
    
    configsess= tf.ConfigProto(allow_soft_placement=True)
    configsess.gpu_options.allow_growth = True
    session = tf.Session(config=configsess)
    session.run(tf.global_variables_initializer())
    writer.add_graph(session.graph)

    print('Training and evaluating...')
    start_time = time.time()
    total_batch = 0  
    best_acc_val = 0.0 
    last_improved = 0  
    require_improvement = 200  
    
    flag = False
    for epoch in range(config.num_epochs):
        print('Epoch:', epoch + 1)
        batch_train = dem.batch_iter(users,news_content,content,news_category,category,
                                    batch_size=config.batch_size,max_length=config.num_words_title,
                                    candidate_size=config.candidate_len,click_size=config.click_len,real_num=config.real_num)
        #batch_train = batch_iter(news_train, users_train, batch_size=config.batch_size,max_length=config.num_words_title,candidate_num=config.candidate_len)
        acc=0
        loss=0
        mrr=0
        for click,click_cat, candidate,candidate_cat,real in batch_train:
            # print(click_cat,candidate_cat)
            feed_dict = feed_data_naml(click,click_cat, candidate,candidate_cat, real, config.dropout_keep_prob)
            
            #print(x_batch.shape[0],x_batch.shape[1])
            if total_batch % config.save_per_batch == 0 :
                s = session.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(s, total_batch)

            if ((total_batch % config.print_per_batch == 0)):#and (total_batch!=0)):
                feed_dict[model.keep_prob] = 1.0
                #loss_train, acc_train = session.run([model.loss, model.acc], feed_dict=feed_dict)
                acc_train=acc/config.print_per_batch
                loss_train=loss/config.print_per_batch
                mrr_train=mrr/config.print_per_batch
                acc=0
                loss=0
                mrr=0
                loss_val, acc_val,mrr_val= evaluate_en_naml(session,test,news_content,content,news_category,category)# todo
                
                if acc_val > best_acc_val:
                    best_acc_val = acc_val
                    last_improved = total_batch
                    saver.save(sess=session, save_path=save_path)
                    improved_str = '*'
                else:
                    improved_str = ''

                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6}, Train Loss: {1:>6.4}, Train nDCG: {2:>7.2%},Train Mrr: {3:6.4}' \
                      + ' Val Loss: {4:>6.4}, Val nDCG: {5:>7.2%}, Val Mrr: {6:6.4} Time: {7} {8}'
                print(msg.format(total_batch, loss_train, acc_train,mrr_train,loss_val, acc_val,mrr_val,time_dif, improved_str))
            
            feed_dict[model.keep_prob] = config.dropout_keep_prob
            #res_train = session.run(model.news_encoder.title_attention.attention_query_vector,feed_dict=feed_dict)
            #print(feed_dict)
            #print(res_train)
            #print(feed_dict)
            #session.run(model.optim, feed_dict=feed_dict)  # è¿è¡ä¼å
            _loss,click_prob,optim=session.run([model.loss,model.click_probability,model.optim], feed_dict=feed_dict)
            _mrr=MRR(click_prob)
            _acc=ACC(click_prob)
            loss+=_loss
            acc+=_acc
            mrr+=_mrr
            total_batch += 1
            '''
            if total_batch - last_improved > require_improvement:
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
            '''
            if ((total_batch % (10*config.print_per_batch) == 0)):
                break
        if flag: 
            break    

def MRR(click_probability):
    rank=np.ones(shape=(config.batch_size,1),dtype=int)
    real_max=np.max(click_probability[:,0:config.real_num],axis=1)
    for i in range(config.batch_size):
        for j in range(1,config.candidate_len):
            if(click_probability[i,j]>real_max[i]):
                rank[i]+=1;
    mrr=sum(1.0/rank)/len(rank)
    return mrr[0]

def ACC(click_probability):
    acc=np.zeros(shape=(config.batch_size,1))
    for i in range(config.batch_size):
        rank={}
        for j in range(0,config.candidate_len):
            rank[j]=click_probability[i,j]
        rank,_=zip(*sorted(rank.items(), key = lambda kv:(-kv[1], -kv[0])))
        for j in range(config.candidate_len):
            if(rank[j]<config.real_num):
                acc[i]+=(2**config.candidate_len-1)/math.log(j+2,2)
            #if(rank[j]<config.real_num and j<config.accept_num):
                #acc[i]+=1
    idcg=0.0
    for i in range(config.real_num):
        idcg+=(2**config.candidate_len-1)/math.log(i+2,2)
    return np.mean(acc)/(idcg)
        
        
if __name__ == '__main__':
    #if len(sys.argv) != 2 or sys.argv[1] not in ['train', 'test']:
        #raise ValueError("""usage: python run_cnn.py [train / test]""")

    print('Configuring CNN model...')
    config = ENConfig()
    
    if sys.argv[1] == 'train_en':
        model = Model(config)
        train_en()
    elif sys.argv[1] == 'naml_en':
        model = NAML(config)
        train_en_naml()
    elif sys.argv[1] == 'dkn_en':
        config = DKNConfig()
        model = DKN(config)
        train_en_dkn()
    else:
        config=CHConfig
        if not os.path.exists(vocab_dir):  
            build_vocab(train_dir, vocab_dir, config.vocab_size)
        categories, cat_to_id = read_category()
        words, word_to_id = read_vocab(vocab_dir)
        config.vocab_size = len(words)
        model = Model(config)
        if sys.argv[1] == 'train':
            train()
        else:
            test()