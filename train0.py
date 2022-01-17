# -*- coding: utf-8 -*-

"""
Created on Wed Oct 21 22:49:38 2020

@author: 1952640
"""

import os
import sys
import time
from datetime import timedelta

import numpy as np
import tensorflow as tf
#import tensorflow.compat.v1 as tf
#tf.disable_eager_execution()
#print('aa')
from sklearn import metrics

from model.__init__ import Model,TCNNConfig
from dataset import read_vocab, read_category, batch_iter, process_file, build_vocab

base_dir = '../data'
train_dir = os.path.join(base_dir, 'train.txt')
test_dir = os.path.join(base_dir, 'test.txt')
val_dir = os.path.join(base_dir, 'val.txt')
vocab_dir = os.path.join(base_dir, 'vocab.txt')

save_dir = './checkpoints/final'
save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路

#获取已用时间
def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def feed_data(click, candidate,real, keep_prob):
    feed_dict = {
        model.input_click: click,
        model.input_candidate: candidate,
        #model.real_len: real,
        model.keep_prob: keep_prob#每个元素被保留的概率，防止过度拟
    }
    #字典
    return feed_dict

#评估在某一数据上的准确率和损失
def evaluate(sess, x_, y_):
    
    data_len = len(x_)
    batch_eval = batch_iter(x_, y_,batch_size=config.batch_size,max_length=config.num_words_title,candidate_num=config.candidate_len)
    total_loss = 0.0
    total_acc = 0.0
    count=0
    for click, candidate,real in batch_eval:
        count+=1
        #print(candidate.shape)
        feed_dict = feed_data(click, candidate,real, 1.0)
        loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
        total_loss += loss
        total_acc += acc 

    return total_loss / count, total_acc / count

def train():
    print("Configuring TensorBoard and Saver...")
    # 配置 Tensorboard，重新训练时，请将tensorboard文件夹删除，不然图会覆盖
    tensorboard_dir = './tensorboard/final'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    #训练实时监控
    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy", model.acc)
    merged_summary = tf.summary.merge_all()#自动管理
    writer = tf.summary.FileWriter(tensorboard_dir)

    #配置 Saver
    saver = tf.train.Saver()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("Loading training and validation data...")
    # 载入训练集与验证
    start_time = time.time()
    news_train, users_train = process_file(train_dir, word_to_id, cat_to_id, config.seq_length)
    news_val, users_val = process_file(val_dir, word_to_id, cat_to_id, config.seq_length)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
    
     # 创建session
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    writer.add_graph(session.graph)

    print('Training and evaluating...')
    start_time = time.time()
    total_batch = 0  # 总批
    best_acc_val = 0.0  # 最佳验证集准确�?
    last_improved = 0  # 记录上一次提升批�?
    require_improvement = 200  # 如果超过1000轮未提升，提前结束训�?
    
    flag = False
    for epoch in range(config.num_epochs):
        print('Epoch:', epoch + 1)
        batch_train = batch_iter(news_train, users_train, batch_size=config.batch_size,max_length=config.num_words_title,candidate_num=config.candidate_len)
        acc=0
        loss=0
        for click, candidate,real in batch_train:
            feed_dict = feed_data(click, candidate, real, config.dropout_keep_prob)
            
            #print(x_batch.shape[0],x_batch.shape[1])
            #feed_dict:填充
            if total_batch % config.save_per_batch == 0 :
                # 每多少轮次将训练结果写入tensorboard scalar
                s = session.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(s, total_batch)

            if ((total_batch % config.print_per_batch == 0)):#and (total_batch!=0)):
                # 每多少轮次输出在训练集和验证集上的性能
                #feed_dict={xs:v_xs,ys:v_ys,keep_prob:1}
                feed_dict[model.keep_prob] = 1.0
                #loss_train, acc_train = session.run([model.loss, model.acc], feed_dict=feed_dict)
                acc_train=acc/config.print_per_batch
                loss_train=loss/config.print_per_batch
                acc=0
                loss=0
                loss_val, acc_val = evaluate(session, news_val, users_val)# todo
                
                if acc_val > best_acc_val:
                    # 保存
                    best_acc_val = acc_val
                    last_improved = total_batch
                    saver.save(sess=session, save_path=save_path)
                    improved_str = '*'
                else:
                    improved_str = ''

                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                      + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif, improved_str))
            
            feed_dict[model.keep_prob] = config.dropout_keep_prob
            #res_train = session.run(model.news_encoder.title_attention.attention_query_vector,feed_dict=feed_dict)
            #print(feed_dict)
            #print(res_train)
            #print(feed_dict)
            #session.run(model.optim, feed_dict=feed_dict)  # 运行优化
            _loss,_acc,optim=session.run([model.loss,model.acc,model.optim], feed_dict=feed_dict)
            loss+=_loss
            acc+=_acc
            total_batch += 1

            if total_batch - last_improved > require_improvement:
                # 验证集正确率长期不提升，提前结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break  # 跳出循环
        if flag:  # 同上
            break

def test():
    print("Loading test data...")
    start_time = time.time()
    x_test, y_test,contents = test,process_file(test_dir, word_to_id, cat_to_id, config.seq_length)

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)  # 读取保存的模�?
    '''
    print('Testing...')
    loss_test, acc_test = evaluate(session, x_test, y_test)
    msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
    print(msg.format(loss_test, acc_test))
    '''
    
    batch_test = test_batch_iter(x_test, y_test, batch_size=config.batch_size,max_length=config.num_words_title,candidate_num=config.candidate_len)
    count=0
    #click_predict = []  # 保存预测结果
    for click, candidate,real,nolist in batch_test:
        feed_dict = feed_data(click, candidate, real, 1.0)
        click_predict=session.run(model.click_probability, feed_dict=feed_dict)
        print('user : ',y_test[nolist[],)
        break
    
    # click_predict = np.array(np.expand_dims(click_predict,1))
    # click_expect = np.zeros(shape=(count,1), dtype=np.int32)
    # 评估
    # print("Precision, Recall and F1-Score...")
    # print(metrics.classification_report(click_expect, click_predict, target_names=['True']))

    # 混淆矩阵
    # print("Confusion Matrix...")
    # cm = metrics.confusion_matrix(click_expect, click_predict)
    # print(cm)

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
    
if __name__ == '__main__':
    #if len(sys.argv) != 2 or sys.argv[1] not in ['train', 'test']:
        #raise ValueError("""usage: python run_cnn.py [train / test]""")

    print('Configuring CNN model...')
    config = TCNNConfig()
    if not os.path.exists(vocab_dir):  # 如果不存在词汇表，重�?
        build_vocab(train_dir, vocab_dir, config.vocab_size)
    categories, cat_to_id = read_category()
    words, word_to_id = read_vocab(vocab_dir)
    config.vocab_size = len(words)
    model = Model(config)

    if sys.argv[1] == 'train':
        train()
    else:
        test()
