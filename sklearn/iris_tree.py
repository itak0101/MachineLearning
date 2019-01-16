# -*- coding: utf-8 -*-
from sklearn.datasets import load_iris
from sklearn import tree
import matplotlib.pyplot as plt
import pandas as pd
import pydotplus
import os
from sklearn.externals.six import StringIO
from sklearn.metrics import accuracy_score

# pandas dataframeの表示設定 (コンソール表示における省略表記を100カラムまで実施しない)
pd.set_option('display.max_columns', 100)

# irisデータの読み込み
iris = load_iris()

# irisデータをpandasのdataframe形式で取得 (列名はiris.feature_namesを利用)
df = pd.DataFrame(iris.data, columns=iris.feature_names)
print(df.dtypes)
print(df.head())

# dataframeにspecies配列を追加 (追加内容はtarget配列をtarget_namesに射影したもの)
df['species'] = iris.target_names[iris.target]
print(df.dtypes)
print(df.head())

# 決定木を作成
clf = tree.DecisionTreeClassifier(max_depth=3)
clf = clf.fit(iris.data, iris.target)

# 決定木に基づく予測を実行
predicted = clf.predict(iris.data)
print('正解率: ' + str(sum(predicted == iris.target) / len(iris.target)))
print('正解率: ' + str(accuracy_score(iris.target, predicted)))

# 決定木をDOTファイルとして出力(GraphVizで読み込むためのファイル)
tree.export_graphviz(clf, out_file="tree.dot",
                         feature_names=iris.feature_names,
                         class_names=iris.target_names,
                         filled=True, rounded=True)

# 決定木をPNG出力(Graphvizのインストールが必要)
dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data,
                         feature_names=iris.feature_names,
                         class_names=iris.target_names,
                         filled=True, rounded=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('iris_tree.png')
graph.write_pdf('iris_tree.pdf')


# 決定木をDOTファイルとして出力(dot_dataから作成するパターン)
with open('tree2.dot', mode='w') as f:
	f.write(dot_data.getvalue())

# PNGファイルを開く
os.system('iris_tree.png')


#---------------------------------------------------------------------
# Refernce
# 
# [1] Scikit-learn: decision-tree (公式ドキュメント)
# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier
#
# [2] scikit-learn で決定木分析 (日本語解説)
# https://pythondatascience.plavox.info/scikit-learn/scikit-learn%E3%81%A7%E6%B1%BA%E5%AE%9A%E6%9C%A8%E5%88%86%E6%9E%90
#
# [3] いまさら聞けない機械学習の評価関数 (正解率に関する説明)
# https://data.gunosy.io/entry/2016/08/05/115345
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