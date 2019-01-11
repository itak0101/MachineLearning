# -*- coding: utf-8 -*-
from sklearn.datasets import load_iris
from sklearn import tree
import matplotlib.pyplot as plt
import pandas as pd
import pydotplus
from sklearn.externals.six import StringIO

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

# 作成した決定木を可視化
clf = tree.DecisionTreeClassifier(max_depth=1)
clf = clf.fit(iris.data, iris.target)
predicted = clf.predict(iris.data)
sum(predicted == iris.target) / len(iris.target)
tree.export_graphviz(clf, out_file="tree.dot",
                         feature_names=iris.feature_names,
                         class_names=iris.target_names,
                         filled=True, rounded=True)

# 作成した決定木をPDF出力(Graphvizのインストールが必要)
dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("graph.pdf")

#---------------------------------------------------------------------
# End

# PDFファイルの出力が上手くいかない場合 graphviz.pyを編集する必要がある
# 以下の部分に、自分の環境におけるgraphvizのbinフォルダのパスを加える
#
#def find_graphviz():
#    (中略)	
#    for path in (
#            '/usr/bin', '/usr/local/bin',
#            '/opt/local/bin',
#            '/opt/bin', '/sw/bin', '/usr/share',
#            '/Applications/Graphviz.app/Contents/MacOS/'):
#    (後略)
#