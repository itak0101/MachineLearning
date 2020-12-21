# -*- coding: utf-8 -*-
# 指定したフォルダに格納されている全ての静止画に対して、Lobe の LocalAPI を用いて推論を行う
# Lobe Website: https://lobe.ai/

#---------------------------------------------------------------------
# ユーザー設定値
#---------------------------------------------------------------------

# 静止画が格納されているフォルダ
targetFolderPath = './'

# 静止画ファイルの種別
imageType = 'png,jpg'

# LobeのモデルURL (Lobeの上部メニュー File -> Export -> Loacal API -> URLに表示されている文字列)
url = "http://localhost:38100/predict/e490dd14-f74c-4d39-8619-a1b5918f928d"

# 出力ファイルパス
outputFilePath = targetFolderPath + 'PredictWithLobe_LocalAPI.csv'


#---------------------------------------------------------------------
# ライブラリの読み込み
#---------------------------------------------------------------------
import base64
import requests
import json
import os
import sys
import glob


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
targetFolderPath = os.path.abspath(targetFolderPath)
if(os.path.exists(targetFolderPath) == False):
    ErrorEnd ('Error End | 指定された静止画格納フォルダが存在しません: ' + targetFolderPath)

# フォルダ内に格納されている静止画のパスを一覧取得する
imgFilePaths = []
extentions = imageType.split(',')
for i, ext in enumerate(extentions):
    imgFilePaths.extend(glob.glob(targetFolderPath + '/**/*.' + ext, recursive=True))
imgFilePaths = sorted(imgFilePaths)

# フォルダ内に 1つも静止画がなければエラー終了
if(len(imgFilePaths) == 0):
    s = 'Error End | 静止画ファイルが存在しません (' + str(os.path.abspath(targetFolderPath)) + ')'
    ErrorEnd (s)

# 静止画のパスを一つ一つ処理してゆく
with open(outputFilePath, 'w', encoding='UTF-8') as f:
    f.write('TargetFile,Result,ActualClass,PredictedClass,Score\n')
    for i, imgFilePath in enumerate(imgFilePaths):

        # ファイルパス、ファイル名、フォルダ名を取得する
        imgFilePath = os.path.abspath(imgFilePath)
        imgFileName = os.path.basename(imgFilePath)
        imgDirName = os.path.basename(os.path.dirname(imgFilePath))

        # 静止画をバイト型データとして読み取り、base64形式データに変換する
        with open(imgFilePath, 'br') as f1:
            b64_img = base64.b64encode(f1.read())

        # モデルに送信するリクエストの内容を設定する
        # (b64_imgは 0文字目に bという文字列が付与されているので、それを除くために 1文字目から入力する)
        payload = '{"inputs":{"Image":"' + str(b64_img)[1:] + '"}}'

        # モデルにリクエストを送信する
        response = requests.request("POST", url, data=payload)

        # モデルから返却されたデータを取得する (取得したデータは1列目 = 確信度でソートする)
        data = json.loads(response.text)
        keyList = data['outputs']['Labels']
        keyList.sort(key=lambda x: x[1], reverse=True)

        # 推論結果を全て表示する
        #print(keyList)

        # 推論結果を整理する
        classActual = imgDirName
        classPredict = keyList[0][0]
        scorePredict = keyList[0][1]
        isPredictSucceed = False
        if classPredict == classActual :
            isPredictSucceed = True

        # コンソール出力 (最も確信度の高いクラスのクラス名と確信度を出力する)
        print ( str(i+1) + '/' + str(len(imgFilePaths)) + ' : ', imgFilePath, isPredictSucceed, classActual, classPredict, scorePredict)

        # CSV出力 (全てのクラス名と確信度を出力する)
        f.write(str(imgFilePath) + ',' + str(isPredictSucceed) + ',' + str(classActual))
        for j, key in enumerate(keyList):
            f.write(',' + str(key[0]) + ',' + str(key[1]))
        f.write('\n')


print ('--- Process Finished -----------------------------\n\n')

#---------------------------------------------------------------------
# End
