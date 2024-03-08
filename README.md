# vista-models
本プロジェクトは、令和5年度の空き家対策モデル事業の二次募集で採択された「IoTカメラと画像認識AIを用いた空き家解体の施工管理システムの開発」において、解体工事現場での「不安全行動」を検出するモデルなどを開発しました。

## vistaとは
vista - Vision Integrated System for Technical Assessment
vista（遠望）は、遠く離れた施工現場を正確に監視・評価するという私たちのコンセプトを体現しています。このシステムを通じて、施工現場の安全性を向上させることを目指しています。

## 推奨環境
Python3.9.5

## 環境変数
カレントディレクトリに以下を記載した`.env`を配置してください。
```.env
OPENAI_API_KEY = {OPENAI APIキー}
```

## 開発環境構築
事前に`pipenv`をインストールしてください。
```command
pip install pipenv
```

以下のコマンドで仮想環境を構築していください。
```command
pipenv install --dev
pipenv run mmpose_install
```

## データの扱い方法
以下のディレクトリに各々のデータを配置してください。
```ディレクトリ構成
./data
    |- annotation
        |- {label studioでexposeしたjsonファイル}
    |- images
        |- {label studioでマウントしたバケットと同じ構成で画像を配置}
    |- outputs # 各AI処理の中間成果物を格納
        |- 01.format_and_cv
        |- 02.pose_estimation
        |- 03.helmet_detection
        |- 04.highplace_detection
        |- 05.safetybelt_detection
```


## ライセンス
本ソフトウェアのライセンスについては、LICENSEファイルをご参照ください。

## 参考文献
[令和５年度 空き家対策モデル事業の二次募集 募集要項](https://www.mlit.go.jp/report/press/house03_hh_000168.html)
