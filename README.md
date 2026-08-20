# picolor

**日本語** ・ [English](README.en.md)

Raspberry PiとCSIカメラの**12 bit RAW**を使い、映像内の任意位置をLab／Linear RGBで連続測定する実験用システムである。

撹拌中の液体、粉体、大型物、塗膜など、一般的な測色器へ入れにくい試料の色変化をその場で記録できる。

> [!IMPORTANT]
> 研究・試作向けであり、分光器や校正済み測色計の代替ではない。医療、安全、法規、取引証明には使用しないこと。

## できること

| 機能 | 内容 |
|---|---|
| 任意位置の連続測定 | 可動式の`Ref`枠で18%グレーカード、`Target`枠で試料を同時測定 |
| 2種類の色表示 | Lab（`L*`、`a*`、`b*`）とLinear RGBを切り替え |
| 12 bit RAW解析 | センサーに近いRAW信号から色を計算 |
| 色補正 | 色基準（Spyder Checkr）の48色の既知Lab値でカメラの色ずれを補正 |
| 測定監視 | 安定性、照明変化、むら、白飛び、再校正の必要性を表示 |
| 記録 | 時系列CSV、撮影条件、警告、スナップショットを保存 |

## 12 bit RAW

比較対象のUSBカメラが8 bit出力の場合、picolorの入力階調は理論上16倍となる。

| 入力 | RAW画素値の段階数 | 比較 |
|---|---:|---:|
| 8 bit映像 | 256（0〜255） | 1倍 |
| picolorの12 bit RAW | 4096（0〜4095） | 16倍 |

現行コードはRaspberry Pi High Quality Cameraの`SBGGR12`を使う。8 bit化で同じ値へ丸められる前の細かな変化を色計算へ渡せる点が特色である。

`L*`自体が4096段階になるわけではない。12 bitのRGB入力から小数値として計算する。実際の識別精度はセンサーノイズ、照明、露光、反射、校正にも左右される。

## 色基準の自動検出（Spyder Checkr）

- 使用製品：**Datacolor Spyder Checkr 48色モデル**
- 形状：見開き2面のハードケース型
- JAN：`4571380541088`
- 対象外：SpyderCHECKR 24、Spyder Checkr Photo／Video

チャートを画面の水平線へ厳密に合わせる必要はない。picolorが次を自動検出し、48色を正しいLab基準へ対応させる。

- 画面内の位置
- 傾き・回転
- 左右2面
- 48個の色パッチ中心

判定の確信度が不足した場合は、誤った色補正を防ぐため校正を中止する。

## 動作環境

| 項目 | 現行版の条件 |
|---|---|
| 本体 | Raspberry Pi 4 Model B／Raspberry Pi 5（Pi 5の8 GB版で確認） |
| OS | Raspberry Pi OS 64-bit Desktop（Debian 13 Trixie） |
| カメラ | High Quality Camera（IMX477）をCSI接続 |
| RAW入力 | 12 bit `SBGGR12` |
| カメラAPI | Picamera2／libcamera |
| 非対応 | Windows、macOS、USBカメラ、その他のCSIカメラ |

**12 bit RAW対応CSIカメラとRaspberry Piのカメラ環境が必須であり、Windows／macOSでは使用できない。**

> [!NOTE]
> 現行版はRaspberry Pi専用である。一般的なWindows PCやMacにはRaspberry Pi用CSIカメラを直接つなぐ端子とPicamera2環境がない。MIPI CSI-2を持つ組込み機器は存在するが、端子、ドライバ、RAW形式が異なるため、そのままでは動作しない。

## 必要な機材

| 分類 | 機材 | 備考 |
|---|---|---|
| Raspberry Pi | Raspberry Pi 4 Model BまたはRaspberry Pi 5、microSD 32 GB以上、ケース・冷却 | 電源はPi 4が公式15 W、Pi 5が公式27 Wを推奨。Pi 5の8 GB版で確認 |
| カメラ | Raspberry Pi High Quality Camera（IMX477、C/CSマウント） | 現行コードの対象 |
| レンズ | Raspberry Pi 6 mm広角レンズ（CSマウント） | 16 mmレンズも選択可 |
| カメラ配線 | 機種別カメラケーブル、PimoroniのCSI–HDMI中継基板2枚、標準HDMIケーブル、短い15ピンCSIケーブル | Pi 4とPi 5では最初のケーブルが異なる。中継基板はPetit Studios製、現在は取扱終了 |
| 色基準 | 色基準チャート（Spyder Checkr 48色モデル）、18%グレーカード | 実機のグレーカードは銀一シルクグレーカード Ver.2 |
| 撮影環境（小型試料） | [HAKUBA LEDスタジオボックス60（AMZLEDSBX60）](https://www.amazon.co.jp/dp/B0923V3439)、固定具 | 約64×62×63 cm。LEDと背景布3色（白・黒・オレンジ）が付属 |
| 撮影環境（大型試料） | 白色LED、拡散板またはソフトボックス、白背景、固定具 | カメラ・照明・試料を固定 |
| 校正・操作 | レンズキャップ、HDMIモニター、キーボード、マウス | 初期設定と画面操作に使用 |

### カメラ接続

| Raspberry Pi | カメラ端子 | 最初のケーブル |
|---|---|---|
| Pi 4 Model B | 標準15ピン `CAMERA` | Standard–Standard |
| Pi 5 | Mini 22ピン `CAM/DISP` | Standard–Mini |

```text
Raspberry Pi
  → 上表の機種別カメラケーブル
  → CSI–HDMI中継基板
  → 標準HDMIケーブル
  → CSI–HDMI中継基板
  → 15ピンCSIカメラケーブル
  → High Quality Camera
```

> [!CAUTION]
> HDMIケーブルはカメラ信号の延長配線に使う。Piの映像出力端子やモニターへ接続してはならない。配線はPiの電源を切って行うこと。信号線とシールドが正しく結線された短いHDMIケーブルを推奨する。

## セットアップ

動作確認済みOSはRaspberry Pi OS 64-bit Desktop（Debian 13 Trixie）である。

```bash
sudo apt update
sudo apt install -y \
  git python3 python3-numpy python3-opencv python3-pil \
  python3-picamera2 python3-scipy fonts-noto-cjk
```

カメラ映像を確認する。

```bash
rpicam-hello --timeout 5000
```

picolorを取得して起動する。

```bash
git clone https://github.com/Higomon/picolor.git
cd picolor
python3 -u -c "from csi.main import main; main()"
```

## 初回の校正

画面の案内に従い、次の順で実行する。

| 順 | キー | 置くもの | 目的 |
|---:|:---:|---|---|
| 1 | `D` | レンズキャップ | 暗所ノイズ補正 |
| 2 | `F` | 均一な白背景 | 照明むら補正 |
| 3 | `W` | 18%グレーカード | 白バランスと相対基準 |
| 4 | `P` | 見開いた色基準（Spyder Checkr 48色モデル） | 48色による色補正 |

色基準（Spyder Checkr）は水平でなくてもよいが、全体を画面内へ入れ、反射や影でパッチを隠さないこと。横倒し、画面外、強い反射・影では検出に失敗する場合がある。

## 測定

1. `Ref`枠を18%グレーカードへ合わせる。
2. `Target`枠を試料の測定位置へ合わせる。
3. 測定可能表示を待つ。
4. `m`で記録を開始し、`s`で停止する。
5. `q`または`ESC`で終了する。

<details>
<summary><strong>向いている試料と注意点</strong></summary>

| 試料 | 用途 | 注意点 |
|---|---|---|
| 撹拌中の液体 | 反応や調色の追跡 | 泡、容器、映り込みを一定にする |
| 粉体・粒 | 表面色の比較 | 厚さ、ならし方、影をそろえる |
| 大型物 | 固定位置の連続測定 | 距離、角度、周囲光を変えない |
| 塗膜・印刷物 | 指定位置の比較 | 光沢と反射方向をそろえる |
| 時間変化する試料 | 乾燥、退色、反応の記録 | カメラと基準物を動かさない |

</details>

<details>
<summary><strong>操作キーと保存内容</strong></summary>

| キー | 動作 |
|:---:|---|
| `D` / `F` / `W` / `P` | 暗所／照明むら／グレーカード／48色の校正 |
| `V` | グレーカード状態の確認 |
| `Tab` | Lab／Linear RGBの切り替え |
| `m` / `s` | 連続記録の開始／停止 |
| `c` | 条件変更後の再校正 |
| `q` / `ESC` | 終了 |

保存対象は時系列CSV、時刻、露光、ゲイン、基準状態、警告、スナップショット、校正記録である。

- 校正：`calibration/`
- 測定結果：`/home/<user>/picolor/results/`

</details>

<details>
<summary><strong>測定原理と限界</strong></summary>

- `Ref`と`Target`を同じ画像で測り、照明変動の影響を抑える。
- 色基準（Spyder Checkr）を既知Lab値による絶対基準、18%グレーカードを測定中の相対基準として使う。
- 暗所ノイズ、照明むら、白バランスも補正する。
- 波長別スペクトルやUV-Vis吸光度は測れない。
- 光沢、泡、影、周囲光、距離、角度の変化は測定値へ影響する。
- カメラ、照明、距離、絞り、基準配置を変えた場合は再校正が必要である。
- 市販測色計と同等の精度は保証しない。用途ごとに既知試料で検証すること。

</details>

<details>
<summary><strong>技術資料</strong></summary>

- [Raspberry Pi Camera documentation](https://www.raspberrypi.com/documentation/accessories/camera.html)
- [High Quality Camera](https://www.raspberrypi.com/products/raspberry-pi-high-quality-camera/)
- [RAW mode and bit depth](https://www.raspberrypi.com/documentation/computers/camera_software.html#mode)
- [Camera Cable](https://www.raspberrypi.com/products/camera-cable/)
- [Pimoroni CSI–HDMI extension](https://shop.pimoroni.com/products/pi-camera-hdmi-cable-extension)
- [Datacolor Spyder Checkr](https://www.datacolor.jp/camera-solution/spyder-checkr.html)

</details>

## ライセンス

[MIT License](LICENSE)

## 状態

実験・研究用途として開発中である。Raspberry Pi OSやハードウェアの更新により動作が変わる場合がある。

Raspberry Pi、Datacolor、SpyderCHECKRは各権利者の商標である。本プロジェクトは各社の公式製品ではない。
