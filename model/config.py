# -*- coding: utf-8 -*-
class ENConfig(object):
    embedding_dim = 128  # 词向量维
    seq_length = 100  # 序列长度
    num_classes = 10  # 类别
    num_filters = 256  # 卷积核数
    kernel_size = 3  # 卷积核尺
    vocab_size = 50000  # 词汇表大
    attention_size = 64
    
    hidden_dim = 256  # 全连接层神经

    dropout_keep_prob = 0.5  # dropout保留比例
    learning_rate = 2e-4  # 学习

    batch_size = 64  # 每批训练大小
    num_epochs = 10  # 总迭代轮

    print_per_batch = 30  # 每多少轮输出一次结
    save_per_batch = 10  # 每多少轮存入tensorboard
    
    # For additive attention
    query_vector_dim = 128
    num_words_title = seq_length
    
    candidate_len = 10
    click_len = 30
    real_num=3
    accept_num=5
    
    num_attention_heads = 16
    
    model='cnn' 
    rnn_num=3

    category_num = 40
    category_embedding_dim = 64
    
class CHConfig(object):
    embedding_dim = 128  # 词向量维
    seq_length = 100  # 序列长度
    num_classes = 10  # 类别
    num_filters = 256  # 卷积核数
    kernel_size = 3  # 卷积核尺
    vocab_size = 5000  # 词汇表大
    attention_size = 64
    
    hidden_dim = 256  # 全连接层神经

    dropout_keep_prob = 0.5  # dropout保留比例
    learning_rate = 2e-4  # 学习

    batch_size = 64  # 每批训练大小
    num_epochs = 10  # 总迭代轮

    print_per_batch = 30  # 每多少轮输出一次结
    save_per_batch = 10  # 每多少轮存入tensorboard
    
    # For additive attention
    query_vector_dim = 128
    num_words_title = seq_length
    
    candidate_len = 6 
    click_len = 20
    real_num= 2
    accept_num=5
    
    num_attention_heads = 16
    
    model='cnn' 
    rnn_num=1
    
class DKNConfig(object):
    embedding_dim = 128  # 词向量维
    seq_length = 50  # 序列长度
    num_classes = 10  # 类别
    num_filters = 256  # 卷积核数
    kernel_size = 3  # 卷积核尺
    vocab_size = 50000  # 词汇表
    entity_size = 50000
    attention_size = 64
    
    hidden_dim = 256  # 全连接层神经

    dropout_keep_prob = 0.5  # dropout保留比例
    learning_rate = 2e-4  # 学习

    batch_size = 64  # 每批训练大小
    num_epochs = 10  # 总迭代轮

    print_per_batch = 30  # 每多少轮输出一次结
    save_per_batch = 10  # 每多少轮存入tensorboard
    
    # For additive attention
    query_vector_dim = 128
    num_words_title = seq_length
    
    candidate_len = 10
    click_len = 30
    real_num=3
    accept_num=5
    
    num_attention_heads = 16
    
    model='cnn' 
    rnn_num=3

    category_num = 40
    category_embedding_dim = 64
