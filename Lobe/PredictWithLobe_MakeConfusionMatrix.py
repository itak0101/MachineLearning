# -*- coding: utf-8 -*-
# PredictWithLobe_LoacalAPI.py や PredictWithLobe_LoacalAPI.py の出力ファイルからConfusion Matrixを作成する

#---------------------------------------------------------------------
# ユーザー設定値
#---------------------------------------------------------------------

# 入力ファイルパス
inputFilePath = './PredictWithLobe_LocalAPI.csv'

# 出力ファイル
outputFilePath = './ConfusionMatrix.csv'


#---------------------------------------------------------------------
# ライブラリの読み込み
#---------------------------------------------------------------------
import os
import sys
import csv


#---------------------------------------------------------------------
# エラー終了 (コンソールでエラー内容を通知して終了)
#---------------------------------------------------------------------
def ErrorEnd(msg):

    # エラーメッセージを表示
    print('\n' + msg + '\n')

    # プロセス終了
    sys.exit()


#---------------------------------------------------------------------
# メイン
#---------------------------------------------------------------------
print ('\n\n--- Process Start --------------------------------')

# フォルダの存在確認 (動画ファイル格納フォルダ)
inputFilePath = os.path.abspath(inputFilePath)
if(os.path.exists(inputFilePath) == False):
    ErrorEnd ('Error End | 指定された入力ファイルが存在しません: ' + inputFilePath)

# クラス名を格納する配列を定義
ActualClasses = []
PredictedClasses = []

# 入力ファイルを1行ずつ読み取り、 2列目(実際のクラス名)と 3列目(推論結果クラス名)の値を配列に格納してゆく
with open(inputFilePath, "r", encoding="UTF-8") as fr:
    reader = csv.reader(fr, delimiter=",", doublequote=True, lineterminator="\r\n", quotechar='"', skipinitialspace=True)
    for i, row in enumerate(reader):
        if(i == 0): continue
        ActualClass = row[2]
        PredictedClass = row[3]
        ActualClasses.append(ActualClass)
        PredictedClasses.append(PredictedClass)

# 2列目(実際のクラス名)と 3列目(推論結果クラス名)の値をソートして重複を削除する
ActualClasses = set(ActualClasses)
PredictedClasses = set(PredictedClasses)
ActualClasses = sorted(ActualClasses)
PredictedClasses = sorted(PredictedClasses)

# コンソール出力
print('InputFile: ' + str(inputFilePath))
print('OutputFile: ' + str(outputFilePath))
print('ActualClasses: ' + str(ActualClasses))
print('PredictedClasses: '+ str(PredictedClasses))
print('\nConfusion Matrix')

# 出力ファイルを開く
with open(outputFilePath, 'w', encoding='UTF-8') as fw:

    # Confusion Matrixを数えるための配列を定義 (サイズ = 実際のクラス数 × 推論結果クラス数)
    counter2D = []
    for i, Aclass in enumerate(ActualClasses):
        counter1D = []
        for j, Pclass in enumerate(PredictedClasses):
            counter1D.append(0)
        counter2D.append(counter1D)

    # 出力ファイルのヘッダを作成・出力
    sLine1, sLine2= ',', ','
    for j, Pclass in enumerate(PredictedClasses):
        sLine1 = sLine1 + ',' + 'Predict'
        sLine2 = sLine2 + ',' + Pclass
    fw.write(sLine1 + '\n')
    fw.write(sLine2 + '\n')
    print(sLine1)
    print(sLine2)

    # Confusion Matrixを数え上げる
    for i, Aclass in enumerate(ActualClasses):
        for j, Pclass in enumerate(PredictedClasses):
            with open(inputFilePath, "r", encoding="UTF-8") as fr:
                reader = csv.reader(fr, delimiter=",", doublequote=True, lineterminator="\r\n", quotechar='"', skipinitialspace=True)
                for k, row in enumerate(reader):
                    if((Aclass == row[2]) and (Pclass == row[3])):
                        counter2D[i][j] = counter2D[i][j] + 1
    print(counter2D)

    # Confusion Matrixを出力する
    for i, Aclass in enumerate(ActualClasses):
        sLine = 'Actual'
        sLine = sLine + ',' + Aclass
        for j, Pclass in enumerate(PredictedClasses):
            sLine = sLine + ',' + str(counter2D[i][j])
        fw.write(sLine + '\n')
        print(sLine)

print ('--- Process Finished -----------------------------\n\n')

#---------------------------------------------------------------------
# End
