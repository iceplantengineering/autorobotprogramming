# Visual Components アドオン開発の経緯と今後のアクション

## 1. 背景

`autorobotprogramming` プロジェクトの開発継続を目指し、ソースコードの分析から開始しました。

分析の結果、プロジェクトが停滞した原因は、Visual Components（以下、VC）の単一コンポーネントに紐づく「コンポーネントスクリプト」では、レイアウト全体のグローバルな制御が困難である、という設計上の問題にあると特定しました。

## 2. 実施したこと

上記の問題を解決するため、`page_1.html`の情報を元に、コンポーネントスクリプトから**「Pythonアドオン」**へとリファクタリング（再設計）を行いました。

1.  **アドオン構造の採用:** VCアプリケーション全体を制御できるPythonアドオン方式に切り替えました。
2.  **フォルダの作成:** ドキュメントに基づき、アドオン用のフォルダパス (`C:\Users\yamaj\Documents\Visual Components\4.10\My Commands\Python 3`) を作成しました。
3.  **リファクタリング:** `vc_robot_controller.py` のロジックを、グローバル制御が可能なアドオン (`GlobalController.py`) として再実装しました。
4.  **問題のデバッグ:** アドオンがVCに読み込まれない問題が発生したため、以下のデバッグ作業を行いました。
    *   フォルダパス、VCのバージョン（4.9 → 4.10）の確認と修正。
    *   `__init__.py` の構文を公式ドキュメントに合わせて修正。
    *   アドオン用フォルダ名がローカライズ（日本語化）されている可能性を考慮し、確認と修正を試みました。
    *   最終的に、問題を切り分けるため、最小構成の「Hello World」アドオンを作成してテストを行いました。

## 3. 現状の課題

**最小構成の「Hello World」アドオンですらVCに読み込まれず、UIにボタンが表示されない状態です。**

また、VC起動時にPythonに関するエラーメッセージなども一切表示されないことから、問題はアドオンのコード自体ではなく、**Visual Componentsのアプリケーション内部の設定や実行環境に起因する可能性が極めて高い**と結論付けました。

## 4. 今後のアクション

誠に恐縮ですが、これ以上の調査は私（AIアシスタント）のアクセス可能な範囲を超えています。つきましては、ユーザー様ご自身で、Visual Componentsの専門家（公式フォーラムなど）に問い合わせていただく必要がございます。

その際、以下の情報を提供することで、問題が迅速に解決する可能性が高まります。

### 共有すべき情報

**件名:** 「Pythonアドオンが読み込まれない (VC 4.10)」

**本文:**

> Visual Components 4.10 を使用しています。
> Pythonアドオンを追加しようとしていますが、UIにボタンが表示されず、アドオンが読み込まれません。
> 
> **環境:**
> - Visual Components Premium 4.10
> - アドオンパス: `C:\Users\yamaj\Documents\Visual Components\4.10\My Commands\Python 3`
> 
> **状況:**
> - VC起動時にPythonに関するエラーメッセージは一切表示されません。
> - 以下の最小構成の「Hello World」コードを試しましたが、同様にボタンが表示されません。
> 
> 何か確認すべき設定や、考えられる原因はありますでしょうか。

**`__init__.py` のコード:**
```python
from vcApplication import *

def OnAppInitialized():
  cmd_path = getApplicationPath() + "HelloWorld.py"
  loadCommand("HelloWorld", cmd_path)
  addMenuItem("VcTabHome/TestGroup", "SayHello", -1, "HelloWorld")
```

**`HelloWorld.py` のコード:**
```python
from vcCommand import *

def OnStart():
  app = getApplication()
  app.messageBox("Info", "Hello World!", 64, 0)

cmd = getCommand()
cmd.addState(OnStart)
```
