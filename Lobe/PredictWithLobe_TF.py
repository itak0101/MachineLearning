# -*- coding: utf-8 -*-
# 指定したフォルダに格納されている全ての静止画に対して、Lobe の TensorFlow モデルを用いて推論を行う
# Lobe Website: https://lobe.ai/

# 以下の構成になっている前提
#
# Top
#  | example (カレントフォルダ)
#  |  | PredictWithLobe_TF.py
#  |  | ...
#  | variables
#  |  | ...
#  | saved_model.pb
#  | signature.json
#

#---------------------------------------------------------------------
# ユーザー設定値
#---------------------------------------------------------------------

# 静止画が格納されているフォルダ
targetFolderPath = './'

# 静止画ファイルの種別
imageType = 'png,jpg'

# 出力ファイルパス
outputFilePath = targetFolderPath + 'PredictWithLobe_TF.csv'


#---------------------------------------------------------------------
# ライブラリの読み込み
#---------------------------------------------------------------------
import json
import os
import sys
import glob

import argparse
import os
import json
import tensorflow as tf
from PIL import Image
import numpy as np


#---------------------------------------------------------------------
# Skeleton code showing how to load and run the TensorFlow SavedModel export package from Lobe.
# Copyright (c) Microsoft Corporation.  All rights reserved.
#---------------------------------------------------------------------

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..")  # default assume that our export is in this file's parent directory

class Model(object):
    def __init__(self, model_dir=MODEL_DIR):
        # make sure our exported SavedModel folder exists
        model_path = os.path.realpath(model_dir)
        if not os.path.exists(model_path):
            raise ValueError(f"Exported model folder doesn't exist {model_dir}")
        self.model_path = model_path

        # load our signature json file, this shows us the model inputs and outputs
        # you should open this file and take a look at the inputs/outputs to see their data types, shapes, and names
        with open(os.path.join(model_path, "signature.json"), "r") as f:
            self.signature = json.load(f)
        self.inputs = self.signature.get("inputs")
        self.outputs = self.signature.get("outputs")

        # placeholder for the tensorflow session
        self.session = None

    def load(self):
        self.cleanup()
        # create a new tensorflow session
        self.session = tf.compat.v1.Session(graph=tf.Graph())
        # load our model into the session
        tf.compat.v1.saved_model.loader.load(sess=self.session, tags=self.signature.get("tags"), export_dir=self.model_path)

    def predict(self, image: Image.Image):
        # load the model if we don't have a session
        if self.session is None:
            self.load()
        # get the image width and height
        width, height = image.size
        # center crop image (you can substitute any other method to make a square image, such as just resizing or padding edges with 0)
        if width != height:
            square_size = min(width, height)
            left = (width - square_size) / 2
            top = (height - square_size) / 2
            right = (width + square_size) / 2
            bottom = (height + square_size) / 2
            # Crop the center of the image
            image = image.crop((left, top, right, bottom))
        # now the image is square, resize it to be the right shape for the model input
        if "Image" not in self.inputs:
            raise ValueError("Couldn't find Image in model inputs - please report issue to Lobe!")
        input_width, input_height = self.inputs["Image"]["shape"][1:3]
        if image.width != input_width or image.height != input_height:
            image = image.resize((input_width, input_height))
        # make 0-1 float instead of 0-255 int (that PIL Image loads by default)
        image = np.asarray(image) / 255.0
        # create the feed dictionary that is the input to the model
        # first, add our image to the dictionary (comes from our signature.json file)
        feed_dict = {self.inputs["Image"]["name"]: [image]}

        # list the outputs we want from the model -- these come from our signature.json file
        # since we are using dictionaries that could have different orders, make tuples of (key, name) to keep track for putting
        # the results back together in a dictionary
        fetches = [(key, output["name"]) for key, output in self.outputs.items()]

        # run the model! there will be as many outputs from session.run as you have in the fetches list
        outputs = self.session.run(fetches=[name for _, name in fetches], feed_dict=feed_dict)
        # do a bit of postprocessing
        results = {}
        # since we actually ran on a batch of size 1, index out the items from the returned numpy arrays
        for i, (key, _) in enumerate(fetches):
            val = outputs[i].tolist()[0]
            if isinstance(val, bytes):
                val = val.decode()
            results[key] = val
        return results

    def cleanup(self):
        # close our tensorflow session if one exists
        if self.session is not None:
            self.session.close()
            self.session = None

    def __del__(self):
        self.cleanup()


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

# モデルの読み込み
model = Model()
model.load()

# 静止画のパスを一つ一つ処理してゆく
with open(outputFilePath, 'w', encoding='UTF-8') as f:
    print('--- Results of Prediction ---')
    f.write('TargetFile,Result,ActualClass,PredictedClass,Score\n')
    for i, imgFilePath in enumerate(imgFilePaths):

        # ファイルパス、ファイル名、フォルダ名を取得する
        imgFilePath = os.path.abspath(imgFilePath)
        imgFileName = os.path.basename(imgFilePath)
        imgDirName = os.path.basename(os.path.dirname(imgFilePath))

        # 画像ファイルの読み込み
        image = Image.open(imgFilePath)
        if image.mode != "RGB":
            image = image.convert("RGB")

        # 推論実行
        outputs = model.predict(image)

        # 推論結果を全て表示する
        #print(outputs)

        # 推論結果を整理する
        classActual = imgDirName
        classPredict = outputs['Prediction']
        scores = outputs['Confidences']
        scores.sort(reverse=True)
        scorePredict = scores[0]
        isPredictSucceed = False
        if classPredict == classActual :
            isPredictSucceed = True

        # コンソール出力 (最も確信度の高いクラスのクラス名と確信度を出力する)
        print ( str(i+1) + '/' + str(len(imgFilePaths)) + ' : ', imgFilePath, isPredictSucceed, classActual, classPredict, scorePredict)

        # CSV出力 (最も確信度の高いクラスのクラス名と確信度を出力する)
        f.write(str(imgFilePath) + ',' + str(isPredictSucceed) + ',' + str(classActual)+ ',' + str(classPredict)+ ',' + str(scorePredict) + '\n')

print ('--- Process Finished -----------------------------\n\n')

#---------------------------------------------------------------------
# End
