import os

from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import statistics as stats


def real_nonLinear_trainer():
    data_type = 'bvp'
    data1 = np.loadtxt('nonLinear/real/UBFC/data_all_1(-1,1).csv', delimiter=',')[:, [1, 2, 3, 4, 5]]
    data2 = np.loadtxt('nonLinear/real/UBFC/data_all_2(-1,1).csv', delimiter=',')[:, [1, 2, 3, 4, 5]]
    # data3 = np.loadtxt('splict/rp/dim7_tau8_' + data_type + '/0.1/data_all_3(-1,1)(fake_thres).csv', delimiter=',')
    data = np.r_[data1, data2]
    # data = data[:180]
    target = np.loadtxt('target/dim7_tau8_three.csv', delimiter=',')
    target = target[:9856]
    mm = MinMaxScaler(feature_range=(-1, 1))
    data = mm.fit_transform(data)
    num = 7
    trail_num = 88 * 8
    resultListAcc1, resultListAcc2, resultListAcc3, resultListAcc4 = [], [], [], []
    resultListFsc1, resultListFsc2, resultListFsc3, resultListFsc4 = [], [], [], []
    result1, result2, result3, result4, result5 = 0, 0, 0, 0, 0
    F_score1, F_score2, F_score3, F_score4 = 0, 0, 0, 0
    for i in range(num):
        # print(data[:i*30].shape,data[(i+1)*30:].shape)
        # X_train, X_test, y_train, y_test = train_test_split(
        #     data, target, test_size=0.2)
        # i=21
        print("第" + str(i + 1) + "论")
        X_train = np.r_[data[:i * trail_num], data[(i + 1) * trail_num:(i + num) * trail_num], data[(i + num + 1) * trail_num:(i + num * 2) * trail_num], data[(+ num * 2 + 1) * trail_num:]]
        X_test1 = data[i * trail_num:(i + 1) * trail_num]
        X_test2 = np.r_[
            data[(i + num) * trail_num:(i + num + 1) * trail_num], data[(i + num * 2) * trail_num:(i + num * 2 + 1) * trail_num]]
        # X_test3 = data[(i + 40) * 30:(i + 41) * 30]
        X_test = np.r_[X_test1, X_test2]
        y_train = np.r_[target[:i * trail_num], target[(i + 1) * trail_num:(i + num) * trail_num], target[(i + num + 1) * trail_num:(i + num * 2) * trail_num], target[(i + num * 2 + 1) * trail_num:]]
        y_test1 = target[i * trail_num:(i + 1) * trail_num]
        y_test2 = np.r_[
            target[(i + num) * trail_num:(i + num + 1) * trail_num], target[(i + num * 2) * trail_num:(i + num * 2 + 1) * trail_num]]
        # y_test3=target[(i + 40) * 30:(i + 41) * 30]
        y_test = np.r_[y_test1, y_test2]
        predictor_1 = SVC(gamma='scale', C=1.0, decision_function_shape='ovr', kernel='rbf')
        predictor_1.fit(X_train, y_train)
        predict_lable_1 = predictor_1.predict(X_test)
        result1 += np.mean(predict_lable_1 == y_test)
        F_score1 += f1_score(y_test, predict_lable_1, average='macro')
        resultListAcc1.append(np.mean(predict_lable_1 == y_test))
        resultListFsc1.append(f1_score(y_test, predict_lable_1, average='macro'))
        # if i==14:
        #     print(i)
        print("the accurancy_1 is :", np.mean(predict_lable_1 == y_test))
        # predict_lable_1 = predictor_1.predict(X_test1)
        # print("the accurancy_1_1 is :", np.mean(predict_lable_1 == y_test1))
        # predict_lable_1 = predictor_1.predict(X_test2)
        # print("the accurancy_1_2 is :", np.mean(predict_lable_1 == y_test2))
        # print(classification_report(y_test, predict_lable_1))

        predictor_2 = LinearSVC(dual=False)
        predictor_2.fit(X_train, y_train)
        predict_lable_2 = predictor_2.predict(X_test)
        result2 += np.mean(predict_lable_2 == y_test)
        F_score2 += f1_score(y_test, predict_lable_2, average='macro')
        resultListAcc2.append(np.mean(predict_lable_2 == y_test))
        resultListFsc2.append(f1_score(y_test, predict_lable_2, average='macro'))
        print("the accurancy_2 is :", np.mean(predict_lable_2 == y_test))
        # # predict_lable_2 = predictor_2.predict(X_test1)
        # # print("the accurancy_2_1 is :", np.mean(predict_lable_2 == y_test1))
        # # predict_lable_2 = predictor_2.predict(X_test2)
        # # print("the accurancy_2_2 is :", np.mean(predict_lable_2 == y_test2))
        # # print(classification_report(y_test, predict_lable_2))
        #
        predictor_3 = KNeighborsClassifier()
        predictor_3.fit(X_train, y_train)
        predict_lable_3 = predictor_3.predict(X_test)
        result3 += np.mean(predict_lable_3 == y_test)
        F_score3 += f1_score(y_test, predict_lable_3, average='macro')
        resultListAcc3.append(np.mean(predict_lable_3 == y_test))
        resultListFsc3.append(f1_score(y_test, predict_lable_3, average='macro'))
        print("the accurancy_3 is :", np.mean(predict_lable_3 == y_test))
        # # predict_lable_3 = predictor_3.predict(X_test1)
        # # print("the accurancy_3_1 is :", np.mean(predict_lable_3 == y_test1))
        # # predict_lable_3 = predictor_3.predict(X_test2)
        # # print("the accurancy_3_2 is :", np.mean(predict_lable_3 == y_test2))
        # # print(classification_report(y_test, predict_lable_3))
        #
        predictor_4 = LogisticRegression(penalty='l2', max_iter=10000)
        predictor_4.fit(X_train, y_train)
        predict_lable_4 = predictor_4.predict(X_test)
        result4 += np.mean(predict_lable_4 == y_test)
        F_score4 += f1_score(y_test, predict_lable_4, average='macro')
        resultListAcc4.append(np.mean(predict_lable_4 == y_test))
        resultListFsc4.append(f1_score(y_test, predict_lable_4, average='macro'))
        print("the accurancy_4 is :", np.mean(predict_lable_4 == y_test))
        # # predict_lable_4 = predictor_4.predict(X_test1)
        # # print("the accurancy_4_1 is :", np.mean(predict_lable_4 == y_test1))
        # # predict_lable_4 = predictor_4.predict(X_test2)
        # # print("the accurancy_4_2 is :", np.mean(predict_lable_4 == y_test2))
        # # print(classification_report(y_test, predict_lable_4))

    print("")
    # np.savetxt('result/rp/' + data_type + '_svm(0.01).csv', np.array(result), delimiter=',')
    print(data_type + " the accurancy_1 is :", result1 / num, " std: ", np.std(resultListAcc1))
    print(data_type + " the accurancy_2 is :", result2 / num, " std: ", np.std(resultListAcc2))
    print(data_type + " the accurancy_3 is :", result3 / num, " std: ", np.std(resultListAcc3))
    print(data_type + " the accurancy_4 is :", result4 / num, " std: ", np.std(resultListAcc4))
    print("")
    print(data_type + " the Fscore_1 is :", F_score1 / num, " std: ", np.std(resultListFsc1))
    print(data_type + " the Fscore_2 is :", F_score2 / num, " std: ", np.std(resultListFsc2))
    print(data_type + " the Fscore_3 is :", F_score3 / num, " std: ", np.std(resultListFsc3))
    print(data_type + " the Fscore_4 is :", F_score4 / num, " std: ", np.std(resultListFsc4))


# real_nonLinear_trainer()

def fake_nonLinear_trainer(dataset, method):
    c = 1
    ga = 'scale'

    data1 = np.loadtxt('nonLinear/fake/' + dataset + '/data_all_1_' + method + '(-1,1).csv', delimiter=',')[:, [1, 2, 3, 4, 5]]
    data2 = np.loadtxt('nonLinear/fake/' + dataset + '/data_all_2_' + method + '(-1,1).csv', delimiter=',')[:, [1, 2, 3, 4, 5]]
    length = len(data1)

    data = np.r_[data1, data2]
    target0 = np.loadtxt('target/target_0.csv', delimiter=',')[:len(data1)]
    target1 = np.loadtxt('target/target_1.csv', delimiter=',')[:len(data2)]
    target = np.r_[target0, target1]
    mm = MinMaxScaler(feature_range=(-1, 1))
    data = mm.fit_transform(data)
    model_path = '/result/nonLinear/model/fake/' + dataset + '/' + method
    path = os.getcwd()
    savepath = path + model_path
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    predictor_1 = SVC(gamma=ga, C=c, decision_function_shape='ovr', kernel='rbf')
    predictor_1.fit(data, target)
    joblib.dump(predictor_1, savepath + '/pre1(-1,1).model')
    predictor_2 = LinearSVC(dual=False)
    predictor_2.fit(data, target)
    joblib.dump(predictor_2, savepath + '/pre2(-1,1).model')
    predictor_3 = KNeighborsClassifier()
    predictor_3.fit(data, target)
    joblib.dump(predictor_3, savepath + '/pre3(-1,1).model')
    predictor_4 = LogisticRegression(penalty='l2', max_iter=10000)
    predictor_4.fit(data, target)
    joblib.dump(predictor_4, savepath + '/pre4(-1,1).model')

    # for i in range(num):
    #     print("第" + str(i + 1) + "论")
    #     X_train = np.r_[data[:i * trail_num_1], data[(i + 1) * trail_num_1:length + i * trail_num_2], data[length + (i + 1) * trail_num_2:]]
    #     X_test1 = data[i * trail_num_1:(i + 1) * trail_num_1]
    #     X_test2 = data[length + i * trail_num_2:length + (i + 1) * trail_num_2]
    #     X_test = np.r_[X_test1, X_test2]
    #     y_train = np.r_[target[:i * trail_num_1], target[(i + 1) * trail_num_1:length + i * trail_num_2], target[length + (i + 1) * trail_num_2:]]
    #     y_test1 = target[i * trail_num_1:(i + 1) * trail_num_1]
    #     y_test2 = target[length + i * trail_num_2:length + (i + 1) * trail_num_2]
    #     y_test = np.r_[y_test1, y_test2]
    #
    #     # predictor_1 = SVC(gamma='scale', C=1.0, decision_function_shape='ovr', kernel='rbf')
    #     # predictor_1.fit(X_train, y_train)
    #     # joblib.dump(predictor_1, 'result/nonLinear/model/fake/pre1(-1,1).model')
    #     predict_lable_1 = predictor_1.predict(X_test)
    #     result1 += np.mean(predict_lable_1 == y_test)
    #     result.append(np.mean(predict_lable_1 == y_test))
    #     print("the accurancy_1 is :", np.mean(predict_lable_1 == y_test))
    #
    #     # predictor_2 = LinearSVC(dual=False)
    #     # predictor_2.fit(X_train, y_train)
    #     # joblib.dump(predictor_2, 'result/nonLinear/model/fake/pre2(-1,1).model')
    #     predict_lable_2 = predictor_2.predict(X_test)
    #     result2 += np.mean(predict_lable_2 == y_test)
    #     print("the accurancy_2 is :", np.mean(predict_lable_2 == y_test))
    #
    #     # predictor_3 = KNeighborsClassifier()
    #     # predictor_3.fit(X_train, y_train)
    #     # joblib.dump(predictor_3, 'result/nonLinear/model/fake/pre3(-1,1).model')
    #     predict_lable_3 = predictor_3.predict(X_test)
    #     result3 += np.mean(predict_lable_3 == y_test)
    #     print("the accurancy_3 is :", np.mean(predict_lable_3 == y_test))
    #
    #     # predictor_4 = LogisticRegression(penalty='l2', max_iter=10000)
    #     # predictor_4.fit(X_train, y_train)
    #     # joblib.dump(predictor_4, 'result/nonLinear/model/fake/pre4(-1,1).model')
    #     predict_lable_4 = predictor_4.predict(X_test)
    #     result4 += np.mean(predict_lable_4 == y_test)
    #     print("the accurancy_4 is :", np.mean(predict_lable_4 == y_test))

    # print("")
    # print(data_type + " the accurancy_1 is :", result1 / num)
    # print(data_type + " the accurancy_2 is :", result2 / num)
    # print(data_type + " the accurancy_3 is :", result3 / num)
    # print(data_type + " the accurancy_4 is :", result4 / num)

# fake_nonLinear_trainer('UBFC', 'ODELSTM')

def UBFC_nonlinear_trainer(method, acc):
    data_type = 'bvp'
    dataset = 'UBFC'
    data1 = np.loadtxt('nonLinear/real/' + dataset + '/data_all_1(-1,1).csv', delimiter=',')[:, [1, 2, 3, 4, 5]]
    data2 = np.loadtxt('nonLinear/real/' + dataset + '/data_all_2(-1,1).csv', delimiter=',')[:, [1, 2, 3, 4, 5]]
    data = np.r_[data1, data2]
    # data = data[:180]
    target = np.loadtxt('target/real_4928.csv', delimiter=',')
    mm = MinMaxScaler(feature_range=(-1, 1))
    data = mm.fit_transform(data)
    num = 7
    trail_num = 88 * 8
    resultListAcc1, resultListAcc2, resultListAcc3, resultListAcc4 = [], [], [], []
    resultListFsc1, resultListFsc2, resultListFsc3, resultListFsc4 = [], [], [], []
    result1, result2, result3, result4, result5 = 0, 0, 0, 0, 0
    F_score1, F_score2, F_score3, F_score4 = 0, 0, 0, 0
    predictor_1 = joblib.load('result/nonLinear/model/fake/' + dataset + '/' + method + '/pre1(-1,1).model')
    predictor_2 = joblib.load('result/nonLinear/model/fake/' + dataset + '/' + method + '/pre2(-1,1).model')
    predictor_3 = joblib.load('result/nonLinear/model/fake/' + dataset + '/' + method + '/pre3(-1,1).model')
    predictor_4 = joblib.load('result/nonLinear/model/fake/' + dataset + '/' + method + '/pre4(-1,1).model')
    for i in range(num):
        print("第" + str(i + 1) + "论")
        X_test1 = data[i * trail_num:(i + 1) * trail_num]
        X_test2 = np.r_[
            data[(i + num) * trail_num:(i + num + 1) * trail_num], data[(i + num * 2) * trail_num:(i + num * 2 + 1) * trail_num]]

        X_test = np.r_[X_test1, X_test2]
        y_test1 = target[i * trail_num:(i + 1) * trail_num]
        y_test2 = np.r_[
            target[(i + num) * trail_num:(i + num + 1) * trail_num], target[(i + num * 2) * trail_num:(i + num * 2 + 1) * trail_num]]
        y_test = np.r_[y_test1, y_test2]
        predict_lable_1 = predictor_1.predict(X_test)
        result1 += np.mean(predict_lable_1 == y_test)
        F_score1 += f1_score(y_test, predict_lable_1, average='macro')
        resultListAcc1.append(np.mean(predict_lable_1 == y_test))
        resultListFsc1.append(f1_score(y_test, predict_lable_1, average='macro'))
        acc[0].append(np.mean(predict_lable_1 == y_test))

        print("the accurancy_1 is :", np.mean(predict_lable_1 == y_test))

        predict_lable_2 = predictor_2.predict(X_test)
        result2 += np.mean(predict_lable_2 == y_test)
        F_score2 += f1_score(y_test, predict_lable_2, average='macro')
        resultListAcc2.append(np.mean(predict_lable_2 == y_test))
        resultListFsc2.append(f1_score(y_test, predict_lable_2, average='macro'))
        acc[1].append(np.mean(predict_lable_2 == y_test))
        print("the accurancy_2 is :", np.mean(predict_lable_2 == y_test))

        predict_lable_3 = predictor_3.predict(X_test)
        result3 += np.mean(predict_lable_3 == y_test)
        F_score3 += f1_score(y_test, predict_lable_3, average='macro')
        resultListAcc3.append(np.mean(predict_lable_3 == y_test))
        resultListFsc3.append(f1_score(y_test, predict_lable_3, average='macro'))
        acc[2].append(np.mean(predict_lable_3 == y_test))
        print("the accurancy_3 is :", np.mean(predict_lable_3 == y_test))

        predict_lable_4 = predictor_4.predict(X_test)
        result4 += np.mean(predict_lable_4 == y_test)
        F_score4 += f1_score(y_test, predict_lable_4, average='macro')
        resultListAcc4.append(np.mean(predict_lable_4 == y_test))
        resultListFsc4.append(f1_score(y_test, predict_lable_4, average='macro'))
        acc[3].append(np.mean(predict_lable_4 == y_test))
        print("the accurancy_4 is :", np.mean(predict_lable_4 == y_test))

    print("")
    print(data_type + " the accurancy_1 is :", result1 / num, " std: ", np.std(resultListAcc1))
    print(data_type + " the accurancy_2 is :", result2 / num, " std: ", np.std(resultListAcc2))
    print(data_type + " the accurancy_3 is :", result3 / num, " std: ", np.std(resultListAcc3))
    print(data_type + " the accurancy_4 is :", result4 / num, " std: ", np.std(resultListAcc4))

    print(data_type + " the Fscore_1 is :", F_score1 / num, " std: ", np.std(resultListFsc1))
    print(data_type + " the Fscore_2 is :", F_score2 / num, " std: ", np.std(resultListFsc2))
    print(data_type + " the Fscore_3 is :", F_score3 / num, " std: ", np.std(resultListFsc3))
    print(data_type + " the Fscore_4 is :", F_score4 / num, " std: ", np.std(resultListFsc4))


# UBFC_nonlinear_trainer("ODELSTM", [[], [], [], []])

def wesad_nonlinear_trainer(method, acc):
    data_type = 'bvp'
    length_1 = 568
    length_2 = 298
    length_3 = 0
    data1 = np.loadtxt('nonLinear/real/wesad/data_all_1(-1,1).csv', delimiter=',')[:, [0, 1, 2, 3, 4, 5]]  # 0:0.56 1:0.56 2:0.53 3:0.51 4:0.55 5:0.52
    data2 = np.loadtxt('nonLinear/real/wesad/data_all_2(-1,1).csv', delimiter=',')[:, [0, 1, 2, 3, 4, 5]]
    data_cut_1 = np.zeros((0, len(data1[0])))
    data_cut_2 = np.zeros((0, len(data1[0])))
    for i in range(15):
        data_cut_1 = np.r_[data_cut_1, data1[i * length_1 + length_3:i * length_1 + length_2 + length_3]]
        data_cut_2 = np.r_[data_cut_2, data2[i * length_2:(i + 1) * length_2]]
    data1 = data_cut_1
    data2 = data_cut_2

    # data1 = data1[3000:3000+4470]
    # data1 = data1[:870]
    data = np.r_[data1, data2]
    target0 = np.loadtxt('target/target_0.csv', delimiter=',')[:len(data1)]
    target1 = np.loadtxt('target/target_1.csv', delimiter=',')[:len(data2)]
    target = np.r_[target0, target1]
    mm = MinMaxScaler(feature_range=(-1, 1))
    data = mm.fit_transform(data)
    num = 5
    # trail_num_1 = 58
    # trail_num_2 = 58
    trail_num_1 = length_2 * 3
    # trail_num_1 = 298
    trail_num_2 = length_2 * 3
    resultListAcc1, resultListAcc2, resultListAcc3, resultListAcc4 = [], [], [], []
    resultListFsc1, resultListFsc2, resultListFsc3, resultListFsc4 = [], [], [], []
    len1 = len(data1)
    result1, result2, result3, result4, result5 = 0, 0, 0, 0, 0
    F_score1, F_score2, F_score3, F_score4 = 0, 0, 0, 0

    predictor_1 = joblib.load('result/nonLinear/model/fake/wesad/' + method + '/pre1(-1,1).model')
    predictor_2 = joblib.load('result/nonLinear/model/fake/wesad/' + method + '/pre2(-1,1).model')
    predictor_3 = joblib.load('result/nonLinear/model/fake/wesad/' + method + '/pre3(-1,1).model')
    predictor_4 = joblib.load('result/nonLinear/model/fake/wesad/' + method + '/pre4(-1,1).model')
    for i in range(num):
        # print(data[:i*30].shape,data[(i+1)*30:].shape)
        # X_train, X_test, y_train, y_test = train_test_split(
        #     data, target, test_size=0.2)
        # i=21
        print("第" + str(i + 1) + "论")
        X_train = np.r_[data[:i * trail_num_1], data[(i + 1) * trail_num_1:len1 + i * trail_num_2], data[len1 + (i + 1) * trail_num_2:]]
        X_test1 = data[i * trail_num_1:(i + 1) * trail_num_1]
        X_test2 = data[len1 + i * trail_num_2:len1 + (i + 1) * trail_num_2]
        # X_test3 = data[(i + 40) * 30:(i + 41) * 30]
        X_test = np.r_[X_test1, X_test2]
        y_train = np.r_[target[:i * trail_num_1], target[(i + 1) * trail_num_1:len1 + i * trail_num_2], target[len1 + (i + 1) * trail_num_2:]]
        y_test1 = target[i * trail_num_1:(i + 1) * trail_num_1]
        y_test2 = target[len1 + i * trail_num_2:len1 + (i + 1) * trail_num_2]
        # y_test3=target[(i + 40) * 30:(i + 41) * 30]
        y_test = np.r_[y_test1, y_test2]


        predict_lable_1 = predictor_1.predict(X_test)
        result1 += np.mean(predict_lable_1 == y_test)
        F_score1 += f1_score(y_test, predict_lable_1, average='macro')
        resultListAcc1.append(np.mean(predict_lable_1 == y_test))
        resultListFsc1.append(f1_score(y_test, predict_lable_1, average='macro'))
        acc[0].append(np.mean(predict_lable_1 == y_test))
        print("the accurancy_1 is :", np.mean(predict_lable_1 == y_test))



        predict_lable_2 = predictor_2.predict(X_test)
        result2 += np.mean(predict_lable_2 == y_test)
        F_score2 += f1_score(y_test, predict_lable_2, average='macro')
        resultListAcc2.append(np.mean(predict_lable_2 == y_test))
        resultListFsc2.append(f1_score(y_test, predict_lable_2, average='macro'))
        acc[1].append(np.mean(predict_lable_2 == y_test))
        print("the accurancy_2 is :", np.mean(predict_lable_2 == y_test))


        predict_lable_3 = predictor_3.predict(X_test)
        result3 += np.mean(predict_lable_3 == y_test)
        F_score3 += f1_score(y_test, predict_lable_3, average='macro')
        resultListAcc3.append(np.mean(predict_lable_3 == y_test))
        resultListFsc3.append(f1_score(y_test, predict_lable_3, average='macro'))
        acc[2].append(np.mean(predict_lable_3 == y_test))
        print("the accurancy_3 is :", np.mean(predict_lable_3 == y_test))

        predict_lable_4 = predictor_4.predict(X_test)
        result4 += np.mean(predict_lable_4 == y_test)
        F_score4 += f1_score(y_test, predict_lable_4, average='macro')
        resultListAcc4.append(np.mean(predict_lable_4 == y_test))
        resultListFsc4.append(f1_score(y_test, predict_lable_4, average='macro'))
        acc[3].append(np.mean(predict_lable_4 == y_test))
        print("the accurancy_4 is :", np.mean(predict_lable_4 == y_test))


    print("")

    print(data_type + " the accurancy_1 is :", result1 / num, " std: ", np.std(resultListAcc1))
    print(data_type + " the accurancy_2 is :", result2 / num, " std: ", np.std(resultListAcc2))
    print(data_type + " the accurancy_3 is :", result3 / num, " std: ", np.std(resultListAcc3))
    print(data_type + " the accurancy_4 is :", result4 / num, " std: ", np.std(resultListAcc4))

    print(data_type + " the Fscore_1 is :", F_score1 / num, " std: ", np.std(resultListFsc1))
    print(data_type + " the Fscore_2 is :", F_score2 / num, " std: ", np.std(resultListFsc2))
    print(data_type + " the Fscore_3 is :", F_score3 / num, " std: ", np.std(resultListFsc3))
    print(data_type + " the Fscore_4 is :", F_score4 / num, " std: ", np.std(resultListFsc4))


# wesad_nonlinear_trainer("ODELSTM",[[],[],[],[]])


def CLAS_nonlinear_trainer(method, acc):
    data_type = 'bvp'
    length_1 = 48
    length_2 = 320
    length_3 = 0
    dataset = "CLAS"
    data1 = np.loadtxt('nonLinear/real/CLAS/data_all_3(-1,1).csv', delimiter=',')[:, [1, 2, 3, 4, 5]]  # 0:0.56 1:0.56 2:0.53 3:0.51 4:0.55 5:0.52
    data2 = np.loadtxt('nonLinear/real/CLAS/data_all_2(-1,1).csv', delimiter=',')[:, [1, 2, 3, 4, 5]]
    data_cut_1 = np.zeros((0, len(data1[0])))
    data_cut_2 = np.zeros((0, len(data1[0])))
    for i in range(60):
        data_cut_1 = np.r_[data_cut_1, data1[i * length_1:(i + 1) * length_1]]
        data_cut_2 = np.r_[data_cut_2, data2[i * length_2:i * length_2 + length_1 + length_3]]
    data1 = data_cut_1
    data2 = data_cut_2
    data = np.r_[data1, data2]
    target0 = np.loadtxt('target/target_0.csv', delimiter=',')[:len(data1)]
    target1 = np.loadtxt('target/target_1.csv', delimiter=',')[:len(data2)]
    target = np.r_[target0, target1]
    mm = MinMaxScaler(feature_range=(-1, 1))
    data = mm.fit_transform(data)
    # 五折
    num = 5
    trail_num_1 = length_1 * 12
    trail_num_2 = (length_1 + length_3) * 12
    # 留一
    # num = 60
    # trail_num_1 = length_1
    # trail_num_2 = (length_1 + length_3)
    # trail_num_2 = 320
    resultListAcc1, resultListAcc2, resultListAcc3, resultListAcc4 = [], [], [], []
    resultListFsc1, resultListFsc2, resultListFsc3, resultListFsc4 = [], [], [], []
    len1 = len(data1)
    result1, result2, result3, result4, result5 = 0, 0, 0, 0, 0
    F_score1, F_score2, F_score3, F_score4 = 0, 0, 0, 0


    predictor_1 = joblib.load('result/nonLinear/model/fake/' + dataset + '/' + method + '/pre1(-1,1).model')
    predictor_2 = joblib.load('result/nonLinear/model/fake/' + dataset + '/' + method + '/pre2(-1,1).model')
    predictor_3 = joblib.load('result/nonLinear/model/fake/' + dataset + '/' + method + '/pre3(-1,1).model')
    predictor_4 = joblib.load('result/nonLinear/model/fake/' + dataset + '/' + method + '/pre4(-1,1).model')
    for i in range(num):

        print("第" + str(i + 1) + "论")
        X_train = np.r_[data[:i * trail_num_1], data[(i + 1) * trail_num_1:len1 + i * trail_num_2], data[len1 + (i + 1) * trail_num_2:]]
        X_test1 = data[i * trail_num_1:(i + 1) * trail_num_1]
        X_test2 = data[len1 + i * trail_num_2:len1 + (i + 1) * trail_num_2]
        # X_test3 = data[(i + 40) * 30:(i + 41) * 30]
        X_test = np.r_[X_test1, X_test2]
        y_train = np.r_[target[:i * trail_num_1], target[(i + 1) * trail_num_1:len1 + i * trail_num_2], target[len1 + (i + 1) * trail_num_2:]]
        y_test1 = target[i * trail_num_1:(i + 1) * trail_num_1]
        y_test2 = target[len1 + i * trail_num_2:len1 + (i + 1) * trail_num_2]
        # y_test3=target[(i + 40) * 30:(i + 41) * 30]
        y_test = np.r_[y_test1, y_test2]

        predict_lable_1 = predictor_1.predict(X_test)
        result1 += np.mean(predict_lable_1 == y_test)
        F_score1 += f1_score(y_test, predict_lable_1, average='macro')
        resultListAcc1.append(np.mean(predict_lable_1 == y_test))
        resultListFsc1.append(f1_score(y_test, predict_lable_1, average='macro'))
        acc[0].append(np.mean(predict_lable_1 == y_test))
        print("the accurancy_1 is :", np.mean(predict_lable_1 == y_test))
        print("the F1_score is :", f1_score(y_test, predict_lable_1, average='macro'))

        predict_lable_2 = predictor_2.predict(X_test)
        result2 += np.mean(predict_lable_2 == y_test)
        F_score2 += f1_score(y_test, predict_lable_2, average='macro')
        resultListAcc2.append(np.mean(predict_lable_2 == y_test))
        resultListFsc2.append(f1_score(y_test, predict_lable_2, average='macro'))
        acc[1].append(np.mean(predict_lable_2 == y_test))
        print("the accurancy_2 is :", np.mean(predict_lable_2 == y_test))
        print("the F1_score is :", f1_score(y_test, predict_lable_2, average='macro'))

        predict_lable_3 = predictor_3.predict(X_test)
        result3 += np.mean(predict_lable_3 == y_test)
        F_score3 += f1_score(y_test, predict_lable_3, average='macro')
        resultListAcc3.append(np.mean(predict_lable_3 == y_test))
        resultListFsc3.append(f1_score(y_test, predict_lable_3, average='macro'))
        acc[2].append(np.mean(predict_lable_3 == y_test))
        print("the accurancy_3 is :", np.mean(predict_lable_3 == y_test))
        print("the F1_score is :", f1_score(y_test, predict_lable_3, average='macro'))

        predict_lable_4 = predictor_4.predict(X_test)
        result4 += np.mean(predict_lable_4 == y_test)
        F_score4 += f1_score(y_test, predict_lable_4, average='macro')
        resultListAcc4.append(np.mean(predict_lable_4 == y_test))
        resultListFsc4.append(f1_score(y_test, predict_lable_4, average='macro'))
        acc[3].append(np.mean(predict_lable_4 == y_test))
        print("the accurancy_4 is :", np.mean(predict_lable_4 == y_test))
        print("the F1_score is :", f1_score(y_test, predict_lable_4, average='macro'))

        print()

    print("")
    print(data_type + " the accurancy_1 is :", result1 / num, " std: ", np.std(resultListAcc1))
    print(data_type + " the accurancy_2 is :", result2 / num, " std: ", np.std(resultListAcc2))
    print(data_type + " the accurancy_3 is :", result3 / num, " std: ", np.std(resultListAcc3))
    print(data_type + " the accurancy_4 is :", result4 / num, " std: ", np.std(resultListAcc4))

    print(data_type + " the Fscore_1 is :", F_score1 / num, " std: ", np.std(resultListFsc1))
    print(data_type + " the Fscore_2 is :", F_score2 / num, " std: ", np.std(resultListFsc2))
    print(data_type + " the Fscore_3 is :", F_score3 / num, " std: ", np.std(resultListFsc3))
    print(data_type + " the Fscore_4 is :", F_score4 / num, " std: ", np.std(resultListFsc4))

# CLAS_nonlinear_trainer("ODELSTM", [[], [], [], []])

def contrast_trainer():
    methods = ["ODELSTM", "base", "c-rnn-gan", "cGanWithDS", "dcgan", "time_gan", "PPG—WGAN", "LogGAN"]
    for method in methods:
        acc1 = [[], [], [], []]
        acc2 = [[], [], [], []]
        acc3 = [[], [], [], []]
        print("-------------------------" + method + "---------------------------------------------")
        print("      -------------------------UBFC-----------------------------------")
        UBFC_nonlinear_trainer(method, acc1)
        print("      -------------------------wesad----------------------------------")
        wesad_nonlinear_trainer(method, acc2)
        print("      -------------------------CLAS-----------------------------------")
        CLAS_nonlinear_trainer(method, acc3)
        acc = np.c_[np.array(acc1), np.array(acc2), np.array(acc3)]
        np.savetxt("result/Acc/" + method + "_acc_1.csv", acc, delimiter=',')

# contrast_trainer()
