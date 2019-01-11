# -*- coding: utf-8 -*-
from sklearn.datasets import load_iris
from sklearn import linear_model
import matplotlib.pyplot as plt
import pandas as pd

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

# 線形回帰モデルの作成
lin = linear_model.LinearRegression()
X = df[['sepal length (cm)']]
Y = df['sepal width (cm)'].values
lin.fit(X, Y)

# 出来上がったモデルを表示
print('\nLinear model')
a = lin.coef_[0]
b = lin.intercept_
c = lin.score(X, Y)
print('Y = '  + str(a) + ' X + ' + str(b) + ', Corr=' + str(c))

# 出来上がったモデルを利用して予測
test = pd.DataFrame({'X': [2,4,6]}) # テストデータを用意 (Field名:X, 値:2,4,6)
pred = lin.predict(test)            # 予測実行

# 予測結果を表示
print('\nPrediction')
for i in range(len(test)):
	x  = test['X'][i]
	y1 = a * test['X'][i] + b
	y2 = pred[i]
	print('X=' + str(x) + ' Y= ' + str(y1) + ' Y= ' + str(y2))

# 線形回帰の結果を x<=x<=10 の定義域でプロット
xr = [0,10]
plt.title("Linear Regression")
plt.xlabel("X")
plt.ylabel("Y")
plt.plot(xr, lin.coef_ * xr + lin.intercept_)
plt.scatter(X,Y)
plt.show()

#---------------------------------------------------------------------
# End