# -*- coding: utf-8 -*-
import pandas as pd
from sklearn import svm


#### 学習データの処理 ###############################################################

## 学習データの読み込み
dfTrain = pd.read_csv('train.csv')

# 学習データの前処理1 (Sex列の値を置換。female→1, male→0)
dfTrain['Sex'] = dfTrain['Sex'].str.replace('female', '1') # Sex列のfemaleを1に置換
dfTrain['Sex'] = dfTrain['Sex'].str.replace('male',   '0') # Sex列のmaleを0に置換

# 学習データの前処理2 (主要列のみに絞り込んだ上で、1つでも空欄のある行は削除する)
dfTrain = dfTrain[['Survived','Pclass', 'Sex', 'Age', 'SibSp', 'Parch']]
dfTrain = dfTrain.dropna()

# 学習データの前処理3 (説明変数と目的変数を分離する)
dfTrainFeature = dfTrain[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']]
dfTrainTarget =  dfTrain['Survived']


#### 予測データの処理 ###############################################################

# 予測用データの読み込み
dfTest = pd.read_csv('test.csv')

# 予測用データの前処理1 (Sex列の値を置換。female→1, male→0)
dfTest['Sex'] = dfTest['Sex'].str.replace('female', '1') # Sex列のfemaleを1に置換
dfTest['Sex'] = dfTest['Sex'].str.replace('male',   '0') # Sex列のmaleを0に置換

# 予測用データの前処理2 (Age列の値を置換。空欄→学習データの中央値)
dfTest["Age"] = dfTest["Age"].fillna(dfTrain['Age'].mean())

# 予測用データの前処理3 (説明変数と目的変数を分離する)
dfTestFeature = dfTest[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']]


#### SVM(Suport Vecter Machine) による予測実行 ######################################

# SVM の定義
clf = svm.SVC()

# SVM の作成
clf = clf.fit(dfTrainFeature, dfTrainTarget)

# 学習データにおける正解率を出力
print('正解率: ' + str(clf.score(dfTrainFeature, dfTrainTarget)))

# SVM を使って予測を実行
pTest = clf.predict(dfTestFeature)

# 予測データをCSV出力
dfTestTarget = pd.DataFrame(index=[], columns=['PassengerId', 'Survived'])
dfTestTarget['PassengerId'] = dfTest['PassengerId']
dfTestTarget['Survived'] = pTest
dfTestTarget.to_csv('Titanic_SVM_predicted.csv', index=False)


#---------------------------------------------------------------------
# Refernce
# 
# [1] Scikit-learn: Support Vector Machine (公式ドキュメント)
# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
#
# [2] scikit-learnでSVMのパラメータを調節してみた話 (引数に関する説明)
# https://qiita.com/arata-honda/items/8d08f31aa7d7cbae4c91
#
# [3] 今さら聞けないカーネル法とサポートベクターマシン (アルゴリズムに関する説明)
# https://www.slideshare.net/ShinyaShimizu/ss-11623505
#
# [4] matplotlib基礎 | figureやaxesでのグラフのレイアウト
# https://qiita.com/tsuruokax/items/90167693f142ebb55a7d
#
# [5] Logistic Regression 3-class Classifier (描画に関する説明)
# https://scikit-learn.org/stable/auto_examples/linear_model/plot_iris_logistic.html#sphx-glr-auto-examples-linear-model-plot-iris-logistic-py
# https://shirakonotempura.hatenablog.com/entry/2018/12/19/020123
#
# [6] 機械学習の分類結果を可視化！決定境界
# http://www.dskomei.com/entry/2018/03/04/125249
#
#---------------------------------------------------------------------
# Memo
#
#
#---------------------------------------------------------------------