# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 19:19:59 2021

@author: 1952640
"""

import numpy as np
import tensorflow.keras as kr
import pandas as pd
import random

def news_process(news):
    content=[]
    for i in range(len(news)):
        content_one=news.title[i].strip('[').strip(']').split(',')
        content_one.extend(news.abstract[i].strip('[').strip(']').split(','))
        content.append(list(map(int,content_one)))
    news_content=dict(zip(list(news.id),content))
    return news_content,content

def batch_iter(users,news_id,news,batch_size=64,max_length=100,click_size=30,candidate_size=10,real_num=3):
    batch_count=0
    click_batch=[]
    candidate_batch=[]
    for i in range(len(users)):
        if(batch_count==batch_size):
            batch_count=0
            click_batch=[]
            candidate_batch=[]
        click_id=str(users.history.iloc[i]).split(" ")
        click=[]
        for j in range(len(click_id)):
            if(click_id[j] in news_id):
                click.append(news_id[click_id[j]])
            else:
                j-=1
        if(len(click)<candidate_size+real_num):
            continue
        batch_count+=1
        candidate=click[-real_num:]
        for j in range(candidate_size-real_num):
            candidate.append(news[random.randint(0,120958)])
        click= kr.preprocessing.sequence.pad_sequences(click[0:20],max_length)
        if(len(click)<click_size):
            pad=np.zeros(shape=(click_size-len(click),max_length),dtype=np.int)
            click=np.concatenate((click,pad))
        click_batch.append(click)
        candidate_batch.append( kr.preprocessing.sequence.pad_sequences(candidate,max_length))
        if(batch_count==batch_size):
            yield click_batch,candidate_batch,candidate_batch[0]
    return
        

        