import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report, precision_score, recall_score
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.externals import joblib

from keras.models import Sequential,Model
from keras.layers import Dense,LSTM,Dropout,Conv1D,MaxPool1D,Bidirectional
from keras.optimizers import Adam
from keras import backend as K
from keras.activations import softmax,sigmoid
from keras.utils.np_utils import to_categorical
from keras.losses import CategoricalCrossentropy
import tensorflow as tf


def focal_loss(y_true, y_pred, alpha=0.25, gamma=2):
    """
    focal loss for multi-class classification
    fl(pt) = -alpha_t*(1-pt)^(gamma)*log(pt)
    """
    # To avoid divided by zero
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
    # Calculate p_t
    p_t = tf.where(K.equal(y_true, 1), y_pred, 1-y_pred)
    # Calculate alpha_t
    alpha_t = tf.where(K.equal(y_true, 1), alpha, 1-alpha)
    # Calculate cross entropy
    cross_entropy = -K.log(p_t)
    weight = alpha_t * K.pow((1-p_t), gamma)
    # Calculate focal loss
    loss = weight * cross_entropy
    # Sum the losses in mini_batch
    loss = K.sum(loss, axis=1)

    return loss

def establish_model(train_x,train_y,test_x,test_y,valid_x,valid_y):
    model = Sequential()
    model.add(Conv1D(filters=10,kernel_size=3,strides=1,padding='same',activation='relu'))
    model.add(MaxPool1D(pool_size=5,name = 'CNN'))
    model.add(Bidirectional(LSTM(units = 50,return_sequences=True),merge_mode='concat',name = 'Bi-LSTM1'))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(units = 50,return_sequences=False),merge_mode='concat',name = 'Bi-LSTM2'))
    model.add(Dropout(0.2))
    model.add(Dense(units = 512,activation = "relu"))
    model.add(Dropout(0.2))
    model.compile(loss=focal_loss,optimizer=Adam(), metrics=['accuracy'])
    model.add(Dense(units=1, activation='sigmoid'))

    # train model
    model, pre_y, valid_loss, valid_acc, train_loss, train_acc = train(
        model, train_x, train_y, test_x, test_y, valid_x, valid_y)

    return model, pre_y, valid_loss, valid_acc, train_loss, train_acc


def train(model,train_x,train_y,test_x,test_y,valid_x,valid_y):
    model.fit(train_x,train_y,batch_size=25,epochs=30,class_weight='auto',validation_data=(valid_x,valid_y))
    valid_loss = model.history.history['val_loss']
    valid_acc = model.history.history['val_accuracy']
    train_loss = model.history.history['loss']
    train_acc = model.history.history['accuracy']
    pre_y = model.predict(test_x)
    result = model.evaluate(train_x, train_y)
    print("训练集准确率为:", result[1])
    result = model.evaluate(test_x, test_y)
    print("测试集准确率为:", result[1])

    return model, pre_y, valid_loss, valid_acc, train_loss, train_acc


def data_processing(train_data,test_data,valid_data):
    train_x, train_y = np.array(train_data)[:, :130], np.reshape(np.array(train_data.iloc[:, -1]),(-1,1))
    test_x, test_y = np.array(test_data)[:, :130], np.reshape(np.array(test_data.iloc[:, -1]),(-1,1))
    valid_x, valid_y = np.array(valid_data)[:, :130], np.reshape(np.array(valid_data.iloc[:, -1] ),(-1,1))

    # transfer x shape
    train_x = np.reshape(train_x,(train_x.shape[0],train_x.shape[1],1))
    test_x = np.reshape(test_x,(test_x.shape[0],test_x.shape[1],1))
    valid_x = np.reshape(valid_x,(valid_x.shape[0],valid_x.shape[1],1))

    return train_x, train_y, test_x, test_y, valid_x, valid_y


def drawer(valid_loss, valid_acc, train_loss, train_acc):
    """draw train figure"""
    fig = plt.figure(figsize=(20, 8), dpi=80)
    plt.subplot(121)
    plt.plot(train_loss)
    plt.plot(valid_loss)
    plt.title('loss变化图')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['training', 'validation'], loc='upper right')

    plt.subplot(122)
    plt.plot(train_acc)
    plt.plot(valid_acc)
    plt.title('accuracy变化图')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['training', 'validation'], loc='lower right')

    plt.show()


if __name__ == '__main__':
    # load data
    is_generative = input('选择使用生成的平衡数据还是原非平衡数据?原非平衡数据则输入0,生成的平衡数据则输入1:')
    dataset_choise = "generative" if is_generative == '1' else "original"
    train_data = pd.read_csv('./data/{} training data.csv'.format(dataset_choise))
    test_data = pd.read_csv('./data/{} test data.csv'.format(dataset_choise))
    valid_data = pd.read_csv('./data/{} valid data.csv'.format(dataset_choise))


    train_x, train_y, test_x, test_y, valid_x, valid_y = data_processing(train_data, test_data, valid_data)
    model, pre_y, valid_loss, valid_acc, train_loss, train_acc = establish_model(train_x,train_y,test_x,test_y,
                                                                                 valid_x,valid_y)
    drawer(valid_loss, valid_acc, train_loss, train_acc)

    # # save model
    joblib.dump(model, "model/串行模型.pkl")