# -*- coding: utf-8 -*-
from sklearn import cluster
import pandas as pd


#### 学習データの処理 ###############################################################

# 学習データの読み込み
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


#### k近傍法による予測実行 ##########################################################

# クラスタ数の設定
clustCount = 5

# k-近傍法の設定 (クラスタ数2, 4スレッドで並列計算)
clf = cluster.KMeans(n_clusters=clustCount, n_jobs=4)

# k-近傍法の実行 (実行結果=クラスタ番号をGroup列に格納)
prd = clf.fit_predict(dfTrain)
dfTrain['Group'] = prd

# 予測結果列(初期値:-1)を追加
dfTrain['Predict'] = -1
print(dfTrain.head())

# 各クラスタの概要情報を出力
print('各クラスタに属するレコード数')
print(dfTrain['Group'].value_counts())
for i in range(clustCount):
	print('\n--クラスタ' + str(i) + 'の情報--')
	print(dfTrain[dfTrain['Group']==i].mean())

# 各クラスのSurvivedの平均値が0.5以上ならAlive、0.5以下ならDeadとする
for i in range(i):
	Mean= (dfTrain[dfTrain['Group']==i])['Survived'].mean()
	if Mean < 0.5:
		dfTrain.loc[dfTrain['Group'] == i, 'Predict'] = 0
	else:
		dfTrain.loc[dfTrain['Group'] == i, 'Predict'] = 1

# 正答率の出力
print('正解率: ' + str(sum(dfTrain['Survived'] == dfTrain['Predict']) / len(dfTrain)))

#教師データと予測データを連結して、同じグループにいるデータの生死を判定する


#---------------------------------------------------------------------
# Refernce
# 
# [1] Scikit-learn: decision-tree (公式ドキュメント)
# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
#
# [2] scikit-learn でクラスタ分析 (日本語解説)
# https://pythondatascience.plavox.info/scikit-learn/%E3%82%AF%E3%83%A9%E3%82%B9%E3%82%BF%E5%88%86%E6%9E%90-k-means
#
#---------------------------------------------------------------------
# Memo
#
# [1] このアルゴリズムの場合、クラスタ数を増加させると、1クラスタ 1データとなり確実に正解してしまう。
#
# [2] 教師データと予測データを連結してクラスタ分割し、
#     同クラスタに存在する教師データの値をもとに予測データを判定する方法も考えられる。
#     しかし予測データしか存在しないクラスタはどのように判定するかなど、方針検討する必要がある
#
#---------------------------------------------------------------------