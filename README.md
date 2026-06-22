# 🎨 Stable Diffusion img2img による手書き画像のスタイル変換

> 手書きイラストを入力として、複数スタイルへの変換・パラメータ影響の定量的検証を行ったプロジェクト  
> PyTorch × HuggingFace Diffusers × Google Colab (T4 GPU)

---

## 📌 プロジェクト概要

Stable Diffusion（`CompVis/stable-diffusion-v1-4`）のimg2img機能を用いて、**手書きイラストを起点に複数の画像スタイルへ変換**するパイプラインを構築しました。

単に画像を生成するだけでなく、`strength`・`guidance_scale` の2つのパラメータが生成結果に与える影響を**系統的に実験・考察**することを重視しました。

---

## 🗂️ ディレクトリ構成

```
.
├── 1-StableDiffusion.ipynb   # メインのColabノートブック
├── images/
│   ├── init_image.png        # 入力：512×512に整形した手書き画像
│   └── output/               # 各実験の生成画像
└── README.md
```

---

## 🛠️ 技術スタック

| カテゴリ | 使用技術 |
|---|---|
| 生成モデル | Stable Diffusion v1-4（img2img） |
| フレームワーク | HuggingFace Diffusers, PyTorch |
| 実行環境 | Google Colab（GPU: T4） |
| 画像処理 | PIL / ImageOps |
| 言語 | Python 3.10 |

---

## ⚙️ パイプライン構成

```
手書き画像（任意サイズ）
       │
       ▼
pad_to_square() ── 白背景でパディング → 512×512にリサイズ
       │
       ▼
VAE Encoder（潜在空間へ変換）
       │
       ▼
ノイズ付加（strength で制御）
       │
       ▼
UNet × num_inference_steps 回のデノイズ
（テキストプロンプト + guidance_scale で条件付け）
       │
       ▼
VAE Decoder（画像空間へ復元）
       │
       ▼
生成画像
```

### 前処理：pad_to_square()

手書き画像は縦横比がバラバラなため、白背景の正方形キャンバスに中央配置してから 512×512 にリサイズする前処理を独自実装しました。

```python
def pad_to_square(image):
    w, h = image.size
    max_side = max(w, h)
    padded = Image.new("RGB", (max_side, max_side), (255, 255, 255))
    padded.paste(image, ((max_side - w) // 2, (max_side - h) // 2))
    return padded.resize((512, 512))
```

---

## 🔬 実験内容

### 実験1：strength パラメータのスイープ

`strength` を 0.2 〜 1.0（0.2 刻み）で変化させ、元画像の保持度と変換度のトレードオフを検証。

```python
for k in range(5):
    result = pipe(
        prompt="A photorealistic domestic cat, studio lighting, detailed fur",
        image=init_image,
        strength=0.2 * (k + 1),  # 0.2, 0.4, 0.6, 0.8, 1.0
        num_inference_steps=50,
        guidance_scale=7.5,
    )
```

| strength | 傾向 |
|---|---|
| 0.2 | 元画像にほぼ忠実、変換が弱い |
| 0.4〜0.6 | 構造を保ちながらスタイルが反映 |
| 0.8〜1.0 | 元画像の構造が大きく失われる |

### 実験2：guidance_scale パラメータのスイープ

`guidance_scale` を 3〜15（3 刻み）で変化させ、プロンプトへの追従強度を検証。

```python
for k in range(5):
    result = pipe(
        prompt="A photorealistic domestic cat, studio lighting, detailed fur",
        image=init_image,
        strength=0.75,
        guidance_scale=3 * (k + 1),  # 3, 6, 9, 12, 15
        num_inference_steps=50,
    )
```

### 実験3：スタイル変換の比較

同じ入力画像に対して、異なるスタイルのプロンプトで生成結果を比較。

| スタイル | プロンプト例 |
|---|---|
| リアル写真風 | `"a photorealistic orange tabby cat, studio lighting, sharp focus, 4k"` |
| カートゥーン | `"a cute cartoon cat illustration, simple, colorful"` |
| 油絵 | `"oil painting, thick brushstrokes, impressionist style, vivid colors"` |
| ゴッホ風 | `"in the style of Van Gogh, swirling brushstrokes, vivid colors"` |

---

## 🔍 工夫した点

- **前処理の独自実装**：任意サイズの手書き画像を白背景パディングで 512×512 に整形し、モデルの入力仕様に合わせた
- **パラメータの系統的スイープ**：`strength` と `guidance_scale` を独立して変化させることで、それぞれの効果を切り分けて分析
- **プロンプトエンジニアリングの比較**：短いプロンプト（`"a cat"`）と詳細なプロンプト（スタイル・照明・品質ワード指定）で生成品質の差を検証

---

## 📝 考察・学んだこと

- `strength` と `guidance_scale` は独立して効果があり、**元画像への忠実度**と**プロンプトへの追従度**をそれぞれ制御できる
- 詳細なプロンプト（照明・テクスチャ・品質ワード）を加えると生成品質が明確に向上する
- VAE・UNet・CLIPテキストエンコーダそれぞれの役割を、実験結果と照らし合わせながら理解できた

**今後やりたいこと**
- CLIP score・SSIM による定量評価の導入
- ControlNet を用いた構造保持生成への拡張
- 文書画像（名刺・請求書）への応用可能性の検討

---

## 👤 Author

神戸大学 工学部 情報知能工学科 3年  
**興味領域：** MLエンジニアリング / 画像認識 / 自然言語処理
