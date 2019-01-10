# -*- coding: utf-8 -*-
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# irisデータの読み込み
iris = load_iris()

# irisデータの全体をコンソール出力
#print(iris)

# irisデータのKeyを出力
#print('\n\n-- iris.keys() --')
#print(iris.keys())

# irisデータをKeyごとにコンソール出力
#print('\n\n-- iris.data --')
#print(iris.data)
#print('\n\n-- iris.target --')
#print(iris.target)
#print('\n\n-- iris.target_names --')
#print(iris.target_names)
#print('\n\n-- iris.DESCR --')
#print(iris.DESCR)
#print('\n\n-- iris.feature_names --')
#print(iris.feature_names)
#print('\n\n-- iris.filename --')
#print(iris.filename)
#print(iris)

# irisデータをKeyごとにファイル出力
with open('iris.txt', mode='w') as f:
	print('iris dataset', file=f)
	print('\n-- iris.data --', file=f)
	print(iris.data, file=f)
	print('\n-- iris.target --', file=f)
	print(iris.target, file=f)
	print('\n-- iris.target_names --', file=f)
	print(iris.target_names, file=f)
	print('\n-- iris.DESCR --', file=f)
	print(iris.DESCR, file=f)
	print('\n-- iris.feature_names --', file=f)
	print(iris.feature_names, file=f)
	print('\n-- iris.filename --', file=f)
	print(iris.filename, file=f)

# pandas dataframeの表示設定
#pd.set_option('display.max_columns', 100)

# irisデータをpandasのdataframe形式で取得
df = pd.DataFrame(iris.data, columns=iris.feature_names)
print(df.dtypes)
print(df.head())

# dataframeにspecies配列を追加 (追加内容はtarget配列をtarget_namesに射影したもの)
df['species'] = iris.target_names[iris.target]
print(df.dtypes)
print(df.head())

# dataframeをCSV出力
df.to_csv("iris.csv")

# 統計量を出力
df.describe().to_csv('iris_describe.csv')

# dataframeからデータを切り出す(品種ごと)
df_A = df[df['species'] == 'setosa']
df_B = df[df['species'] == 'versicolor']
df_C = df[df['species'] == 'virginica']

# 散布図行列の画像出力
sns.pairplot(df  , hue='species', diag_kind='kde').savefig('iris_pairplot.png')
sns.pairplot(df_A, hue='species', diag_kind='kde', kind='reg').savefig('iris_pairplot-A.png')
sns.pairplot(df_B, hue='species', diag_kind='kde', kind='reg').savefig('iris_pairplot-B.png')
sns.pairplot(df_C, hue='species', diag_kind='kde', kind='reg').savefig('iris_pairplot-C.png')

# ピアソンの相関係数の行列を作成
df_A.corr(method='pearson').to_csv('iris_corr_A.csv')
df_B.corr(method='pearson').to_csv('iris_corr_B.csv')
df_C.corr(method='pearson').to_csv('iris_corr_C.csv')



#---------------------------------------------------------------------
# End
