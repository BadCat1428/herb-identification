# 智能中草药识别系统

基于图像识别的中草药识别工具，支持**离线模型推理**与**在线 API 识别**双通道，并集成 AI 问答辅助学习。

> 本项目为中小学计算思维类参赛作品，代码开源以便评审。

---

## 功能特性

| 功能 | 说明 |
|---|---|
| **三种识别模式** | 自动 / 在线 / 离线，界面右上角切换 |
| **离线推理** | 本地 TFLite 模型（35 类），完全断网可用 |
| **在线识别** | 百度智能云植物识别 API，准确率更高 |
| **资料整合** | 识别后自动关联百度百科简介，无需另开浏览器 |
| **AI 问答** | 结合当前识别结果，向大模型追问（如"当归孕妇能吃吗"） |
| **置信度提示** | 可调阈值，结果可信度不足时主动提示 |

---

## 快速开始

### 方式一：直接运行 exe（推荐给普通用户）

下载 Release 中的 `中草药识别.exe`，双击运行，无需安装 Python。

- 首次启动约 5~10 秒（加载离线模型，非卡死）
- 需要 64 位 Windows 10/11

### 方式二：源码运行

```bash
pip install requests pillow numpy pywebview ai-edge-litert
python 中草药识别.py
```

---

## 配置密钥

**本项目不包含任何密钥。** 运行前需自行配置：

1. 复制 `config.example.json` 为 `config.json`
2. 填入自己的密钥

```json
{
  "baidu_api_key": "你的百度 API Key",
  "baidu_secret_key": "你的百度 Secret Key",
  "chat_api_base": "https://api.deepseek.com/v1",
  "chat_api_key": "你的 DeepSeek API Key",
  "chat_model": "deepseek-chat"
}
```

也可用环境变量（优先级高于文件）：

| 环境变量 | 对应配置项 |
|---|---|
| `BAIDU_API_KEY` | 百度 API Key |
| `BAIDU_SECRET_KEY` | 百度 Secret Key |
| `CHAT_API_KEY` | 大模型 API Key |

**`config.json` 已在 `.gitignore` 中排除，不会被提交。**

### 密钥获取

- **百度**：[百度智能云](https://console.bce.baidu.com/ai/) → 图像识别 → 创建应用 → 获取 API Key / Secret Key
- **DeepSeek**：[DeepSeek 开放平台](https://platform.deepseek.com/api_keys)

> 只有用在线识别或 AI 问答才需要密钥；纯离线模式无需任何密钥。

---

## 识别模式说明

| 模式 | 行为 | 适用场景 |
|---|---|---|
| **自动** | 优先本地模型，不可用时自动转在线 | 默认 |
| **在线** | 调用百度 API，准确率高 | 有网、求准确 |
| **离线** | 只用本地模型，永不联网 | 无网环境、演示 |

---

## 技术架构

```
┌─────────────────────────────────────────┐
│  前端  index.html (HTML/CSS/JS)          │
│        通过 window.pywebview.api 通信     │
└────────────────┬────────────────────────┘
                 │ pywebview (WebView2)
┌────────────────▼────────────────────────┐
│  后端  中草药识别.py                      │
│  ├── 离线：ai-edge-litert 推理 TFLite     │
│  ├── 在线：百度植物识别 API               │
│  └── 问答：DeepSeek (OpenAI 兼容接口)     │
└─────────────────────────────────────────┘
```

**技术要点：**

- **离线模型**：MobileNetV2 迁移学习，35 类中草药，约 1800 张训练图片
- **推理引擎**：LiteRT（原 TFLite），不依赖完整 TensorFlow
- **界面**：pywebview + 原生 HTML/CSS/JS，前后端分离
- **图像预处理**：超过 4MB 自动压缩；输入按 0-255 原始像素喂入（模型首层已含 Rescaling）

---

## 文件说明

| 文件 | 说明 |
|---|---|
| `中草药识别.py` | 后端主程序 |
| `index.html` | 前端界面 |
| `herb_model.tflite` | 离线识别模型（35 类） |
| `herb_labels.txt` | 药材类别标签 |
| `config.example.json` | 密钥配置模板 |
| `build.spec` | PyInstaller 打包配置 |

---

## 打包

```bash
pip install pyinstaller
pyinstaller build.spec --noconfirm --clean
```

产物在 `dist/` 下。

---

## 已知限制

- 离线模型仅覆盖 35 类常见中草药，超出范围会给出低置信度结果
- 在线识别受 API 调用配额限制
- 识别结果仅供学习参考，**不能替代专业鉴定**，涉及药用时请务必谨慎
- 仅支持 Windows（依赖 WebView2）

---

## 免责声明

本项目为学习与技术验证用途，识别结果不构成任何医学建议。
