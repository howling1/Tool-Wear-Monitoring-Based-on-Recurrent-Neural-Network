import pandas as pd
import numpy as np
import math
from scipy.fftpack import fft, fftshift, ifft
from scipy.fftpack import fftfreq
import pywt
import pywt.data
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, SimpleRNN
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report, precision_score, recall_score
from sklearn.externals import joblib
import matplotlib.pyplot as plt


def time_feature_extractor(data, sample_point):
    """时域特征提取器"""
    means_list, rms_list, waveform_facs_list, clearance_facs_list = [], [], [], []
    for i in data:
        df_mean = pd.Series(i).mean()
        df_rms = math.sqrt(sum([x ** 2 for x in pd.Series(i)]) / sample_point)
        df_waveform_fac = df_rms / (abs(pd.Series(i)).mean())
        sq_sum = 0
        for j in range(sample_point):
            sq_sum += math.sqrt(abs(i[j]))
        df_clearance_fac = (max(pd.Series(i))) / \
            pow((sq_sum / sample_point), 2)

        means_list.append(df_mean)
        rms_list.append(df_rms)
        waveform_facs_list.append(df_waveform_fac)
        clearance_facs_list.append(df_clearance_fac)

    return means_list, rms_list, waveform_facs_list, clearance_facs_list


def get_ps_array(data, num_fft):
    """获取信号功率谱"""
    ps_list = []
    for i in data:
        Y = fft(i, num_fft)
        Y = np.abs(Y)
        Y[0] = Y[0] / num_fft
        Y[1:] = Y[1:] / (num_fft / 2)
        ps = Y ** 2 / num_fft
        ps = ps[:num_fft // 2]
        ps_list.append(ps)
    ps_array = np.array(ps_list)
    return ps_array


def frequency_features_extractor(ps_array, f):
    """频域特征提取器"""
    msf_list, variance_list = [], []
    for i in ps_array:
        # Mean square frequency
        msf = ((np.sqrt(i) * f).sum()) / (i.sum())
        # Power spectrum variance
        variance = i.var()

        msf_list.append(msf)
        variance_list.append(variance)

    return msf_list, variance_list


def wavelet_alternation(data):
    """小波包分解"""
    sample_num, sample_dim = data.shape
    wp_features = np.zeros((sample_num, 8))
    for i in range(sample_num):
        single_data = data[i, :]
        wp = pywt.WaveletPacket(single_data,wavelet='db3',mode='symmetric',maxlevel=3)  # 小波包三层分解
        # Get the node coefficient of level layer
        aaa = wp['aaa'].data
        aad = wp['aad'].data
        ada = wp['ada'].data
        add = wp['add'].data
        daa = wp['daa'].data
        dad = wp['dad'].data
        dda = wp['dda'].data
        ddd = wp['ddd'].data
        # Get norm of node
        ret1 = np.linalg.norm(aaa, ord=None)
        ret2 = np.linalg.norm(aad, ord=None)
        ret3 = np.linalg.norm(ada, ord=None)
        ret4 = np.linalg.norm(add, ord=None)
        ret5 = np.linalg.norm(daa, ord=None)
        ret6 = np.linalg.norm(dad, ord=None)
        ret7 = np.linalg.norm(dda, ord=None)
        ret8 = np.linalg.norm(ddd, ord=None)
        # obtain 8 nodes combined into eigenvector
        single_feature = [ret1, ret2, ret3, ret4, ret5, ret6, ret7, ret8]
        wp_features[i][:] = single_feature

    return wp_features


def feature_engineer(data, fs):
    """特征工程"""
    # 3等分信号数据
    data_x, data_y = np.array(data)[:, :129], data.iloc[:, -1]
    data_x = np.reshape(data_x, (-1, 43))
    # 提取时域特征
    sample_point = data_x.shape[1]
    means_list, rms_list, waveform_facs_list, clearance_facs_list = time_feature_extractor(data_x, sample_point)
    # 提取频域特征
    num_fft = sample_point
    df = fs / (num_fft)
    f = [df * n for n in range(0, num_fft)]
    f = np.array(f[:num_fft // 2])
    ps = get_ps_array(data_x, num_fft)
    msf_list, variance_list = frequency_features_extractor(
        ps, f)
    # 提取小波包特征
    wp_data = wavelet_alternation(data_x)

    data_x = pd.DataFrame(
        wp_data,
        columns=[
            'band 1 energy',
            'band 2 energy',
            'band 3 energy',
            'band 4 energy',
            'band 5 energy',
            'band 6 energy',
            'band 7 energy',
            'band 8 energy'])
    data_x['mean'] = means_list
    data_x['rms'] = rms_list
    data_x['waveform factor'] = waveform_facs_list
    data_x['clearance factor '] = clearance_facs_list
    data_x['msf'] = msf_list
    data_x['Power spectrum variance'] = variance_list

    # 数据标准化
    transfer = StandardScaler()
    data_x = transfer.fit_transform(data_x)
    # one-hot编码转换
    data_y = np.array(pd.get_dummies(data_y))

    # 构造时间步为3的时间序列
    data_x = np.reshape(np.array(data_x), (-1, 3, 14))

    return data_x, data_y


def establish_model(train_x,train_y,test_x,test_y,valid_x,valid_y):
    """"""
    # 构建网络
    model = Sequential()
    # LSTM网络构建
    model.add(LSTM(input_dim=14,units=50,return_sequences=True))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=512, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(units=2, activation="softmax"))
    # 损失函数，优化器定义
    model.compile(loss='categorical_crossentropy',optimizer=Adam(), metrics=['accuracy'])

    # 训练模型
    model, pre_y, valid_loss, valid_acc, train_loss, train_acc = train(
         model, train_x, train_y, test_x, test_y, valid_x, valid_y)
    return model, pre_y, valid_loss, valid_acc, train_loss, train_acc


def train(model,train_x,train_y,test_x,test_y,valid_x,valid_y):
    model.fit(train_x,train_y,batch_size=50,epochs=20,class_weight='auto',validation_data=(valid_x,valid_y))
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


def evaluation(test_y, pre_y):
    test_y_1d = []
    pre_y_1d = []

    for i in test_y:
        state = 0 if i[0] == 1 else 1
        test_y_1d.append(state)

    for i in pre_y:
        state = 0 if i[0] >= i[1] else 1
        pre_y_1d.append(state)

    print(classification_report(test_y_1d, pre_y_1d, labels=(0, 1), target_names=("正常", "异常")))
    print("LSTM模型的AUC指标：", roc_auc_score(test_y, pre_y))


def drawer(valid_loss, valid_acc, train_loss, train_acc):
    """绘制训练过程"""
    fig = plt.figure(figsize=(20, 8), dpi=80)
    plt.subplot(121)
    plt.plot(train_loss)
    plt.plot(valid_loss)
    plt.title('loss变化图')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')

    plt.subplot(122)
    plt.plot(train_acc)
    plt.plot(valid_acc)
    plt.title('accuracy变化图')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')

    plt.show()


if __name__ == '__main__':
    # load data
    is_generative = input('选择使用生成的平衡数据还是原非平衡数据?原非平衡数据则输入0,生成的平衡数据则输入1:')
    dataset_choise = "generative" if is_generative == '1' else "original"
    train_data = pd.read_csv('./data/{} training data.csv'.format(dataset_choise))
    test_data = pd.read_csv('./data/{} test data.csv'.format(dataset_choise))
    valid_data = pd.read_csv('./data/{} valid data.csv'.format(dataset_choise))

    train_x, train_y = feature_engineer(train_data, fs=10)
    test_x, test_y = feature_engineer(test_data, fs=10)
    valid_x, valid_y = feature_engineer(valid_data, fs=10)
    model, pre_y, valid_loss, valid_acc, train_loss, train_acc = establish_model(
        train_x, train_y, test_x, test_y, valid_x, valid_y)

    evaluation(test_y, pre_y)
    drawer(valid_loss, valid_acc, train_loss, train_acc)

    # # save model
    joblib.dump(model, "model/普通LSTM模型.pkl")
