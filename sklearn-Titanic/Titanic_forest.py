# -*- coding: utf-8 -*-
import pandas as pd
import pydotplus
from sklearn import tree
from sklearn import ensemble
from sklearn.externals.six import StringIO


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


#### Random Forest による予測実行 ###################################################

# Random Forest(決定木の深さは1のパターンから9のパターンまで)を作成し、それを用いて予測を実行する
for i in range(1,10):
	
	# 処理の進捗をコンソールに表示
	print('\n' + 'Random Forest Depth: ' + str(i))

	# Random Forest の定義
	# 設定: 深さ i の決定木 20本の森を作成する。3スレッドで並列計算する。
	clf = ensemble.RandomForestClassifier(max_depth=i, n_estimators=20, n_jobs=3)

	# Random Forest の作成
	clf = clf.fit(dfTrainFeature, dfTrainTarget)

	# 学習データにおける正解率と、各説明変数の重要度を出力
	print('正解率: ' + str(clf.score(dfTrainFeature, dfTrainTarget)))
	for (feature_name, importance) in zip(dfTrainFeature.columns.values, clf.feature_importances_):
		print('重要度: ' + '{0:<10}'.format(feature_name), importance)
	
	# Random Forest を使って予測を実行 (森に属する全ての決定木でデータを判定して多数決する)
	pTest = clf.predict(dfTestFeature)

	# 予測データをCSV出力
	dfTestTarget = pd.DataFrame(index=[], columns=['PassengerId', 'Survived'])
	dfTestTarget['PassengerId'] = dfTest['PassengerId']
	dfTestTarget['Survived'] = pTest
	dfTestTarget.to_csv('Titanic_forest_predicted_' + str(i).zfill(2) + '.csv', index=False)
	
	# Random Forest に属する全ての決定木をPNG出力(Graphvizのインストールが必要)
	# 画像が大量になるので、深さ4のケースのみ出力することにする
	if(i != 4):
		continue
	for j in range(len(clf.estimators_)):
		dot_data = StringIO()
		tree.export_graphviz(clf.estimators_[j], out_file=dot_data,
		                         feature_names=dfTrainFeature.columns.values,
		                         class_names=['Dead','Alive'],
		                         filled=True, rounded=True)
		graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
		graph.write_png('Titanic_forest_' + str(j+1).zfill(2) + '.png')
		

#---------------------------------------------------------------------
# Refernce
# 
# [1] Scikit-learn: RandomForest (公式ドキュメント)
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
#
# [2] データ科学便覧 Scikit-learn によるランダムフォレスト(引数に関する説明)
# https://data-science.gr.jp/implementation/iml_sklearn_random_forest.html
#
# [3] Wikipedia - ランダムフォレスト (アルゴリズムに関する説明)
# https://ja.wikipedia.org/wiki/%E3%83%A9%E3%83%B3%E3%83%80%E3%83%A0%E3%83%95%E3%82%A9%E3%83%AC%E3%82%B9%E3%83%88
#
# [4] 決定木入門編 「ウォーリーを探せ」から考える不純度の考え方 (ジニ係数に関する説明)
# http://www.randpy.tokyo/entry/decision_tree_theory
#
#---------------------------------------------------------------------
# Memo
#
# [A] PNGの出力にはGraphvizのインストールが必要
#     https://www.graphviz.org/
#
# [B] PNGファイルの出力が上手くいかない場合、graphviz.pyを編集する必要がある
#     以下の部分に、自分の環境におけるgraphvizのbinフォルダのパスを加える
#
#    def find_graphviz():
#        (中略)	
#        for path in (
#                '/usr/bin', '/usr/local/bin',
#                '/opt/local/bin',
#                '/opt/bin', '/sw/bin', '/usr/share',
#                '/Applications/Graphviz.app/Contents/MacOS/'):
#        (後略)
#
# [C] 決定木の画像の見かた
#     1行目: 分岐条件 (例 petal length (cm) <= 2.45)
#     2行目: ジニ係数による不純度 (例 geni = 0.01、1-(A群/全体)^2-(B群/全体)^2)
#     3行目: その葉まで到達したデータ数、ここで分類するデータ数 (例 samples =100)
#     4行目: 各郡に属するデータ数 (例 value[50,49,5)ならA群に50個、B群に49個、C群に5個)
#     5行目: その葉に属するデータを何群として判定すべきか (例 class=setosa)
#
#---------------------------------------------------------------------