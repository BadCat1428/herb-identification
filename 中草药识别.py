# -*- coding: utf-8 -*-
"""智能中草药识别系统（pywebview 前后端分离版）
识别/问答/配置逻辑与原 tkinter 版完全一致，仅将界面改为前端 index.html，
通过 pywebview 的 js_api 与前端通信。

Copyright (C) 2026 BadCat1428

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import requests
import base64
from PIL import Image
import io
import os
import sys
import json
import traceback
import webbrowser

import webview  # pywebview

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


# ==================== 密钥加载 ====================
# 密钥不写死在源码里：优先读同目录 config.json，其次读环境变量。
# 这样打包出的 exe 内不含任何密钥，反编译也拿不到。
_DEFAULT_CONFIG = {
    "baidu_api_key": "",
    "baidu_secret_key": "",
    "chat_api_base": "https://api.deepseek.com/v1",
    "chat_api_key": "",
    "chat_model": "deepseek-chat",
}


def _load_config():
    """读取 config.json，缺失项回退到环境变量。"""
    cfg = dict(_DEFAULT_CONFIG)

    cfg_path = get_config_path("config.json")
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, encoding="utf-8") as f:
                data = json.load(f)
            for k in cfg:
                if data.get(k):
                    cfg[k] = str(data[k]).strip()
        except Exception:
            pass

    # 环境变量优先于文件（便于 CI 或临时覆盖）
    env_map = {
        "baidu_api_key": "BAIDU_API_KEY",
        "baidu_secret_key": "BAIDU_SECRET_KEY",
        "chat_api_key": "CHAT_API_KEY",
        "chat_api_base": "CHAT_API_BASE",
        "chat_model": "CHAT_MODEL",
    }
    for key, env in env_map.items():
        v = os.environ.get(env, "").strip()
        if v:
            cfg[key] = v
    return cfg


_CONFIG = None


def get_config():
    """惰性加载（需在 get_config_path 定义之后调用）"""
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = _load_config()
    return _CONFIG


def _cfg(key):
    return get_config().get(key, "")


# =====================================================


def make_tflite_interpreter(model_content):
    """优先用轻量的 LiteRT (约5MB)，缺失则回退完整 TensorFlow"""
    try:
        from ai_edge_litert.interpreter import Interpreter
        return Interpreter(model_content=model_content)
    except ImportError:
        import tensorflow as tf
        return tf.lite.Interpreter(model_content=model_content)


def get_base_dir():
    """程序所在目录：frozen 时为 exe 目录，否则为脚本目录"""
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def get_config_path(name):
    return os.path.join(get_base_dir(), name)


def resource_path(name):
    """前端资源路径：frozen 时在 sys._MEIPASS，否则脚本同目录"""
    if getattr(sys, 'frozen', False):
        meipass = getattr(sys, '_MEIPASS', None)
        if meipass:
            return os.path.join(meipass, name)
    return os.path.join(get_base_dir(), name)


def _find_model_paths():
    """查找离线模型 herb_model.tflite + herb_labels.txt"""
    candidates = []
    if getattr(sys, 'frozen', False):
        candidates.append(os.path.dirname(sys.executable))
        meipass = getattr(sys, '_MEIPASS', None)
        if meipass:
            candidates.append(meipass)
    else:
        candidates.append(os.path.dirname(os.path.abspath(__file__)))
    for d in candidates:
        m = os.path.join(d, "herb_model.tflite")
        l = os.path.join(d, "herb_labels.txt")
        if os.path.exists(m) and os.path.exists(l):
            return m, l
    return None, None


def _log_offline_error(prefix=""):
    """离线失败追加写 offline_error.log"""
    try:
        with open(get_config_path("offline_error.log"),
                  "a", encoding="utf-8") as f:
            f.write(prefix + traceback.format_exc() + "\n")
    except Exception:
        pass


class HerbAPI:
    """pywebview js_api：暴露给前端 window.pywebview.api 的全部方法"""

    def __init__(self):
        self.mode = self.load_mode()
        self.threshold = self.load_threshold()
        # 刚识别出的 top1 药材（用于 AI 问答上下文）
        self.last_top1 = None  # {"name": str, "score": float}
        # 对话历史（发给 AI 的 messages）
        self.chat_messages = []
        # 离线模型预热缓存
        self._interp = None
        self._labels = None
        self._loaded_model_path = None
        self._window = None

    # ---------------- 配置读写（逻辑原样保留） ----------------
    def load_threshold(self):
        try:
            with open(get_config_path("threshold.json"), encoding="utf-8") as f:
                v = float(json.load(f)["threshold"])
            if 0.50 <= v <= 0.95:
                return v
        except Exception:
            pass
        return 0.75

    def save_threshold(self, value):
        try:
            with open(get_config_path("threshold.json"), "w", encoding="utf-8") as f:
                json.dump({"threshold": value}, f)
        except Exception:
            pass

    def load_mode(self):
        try:
            with open(get_config_path("mode.json"), encoding="utf-8") as f:
                m = json.load(f)["mode"]
            if m in ("自动", "在线", "离线"):
                return m
        except Exception:
            pass
        return "自动"

    def save_mode(self):
        try:
            with open(get_config_path("mode.json"), "w", encoding="utf-8") as f:
                json.dump({"mode": self.mode}, f, ensure_ascii=False)
        except Exception:
            pass

    # ---------------- js_api 接口 ----------------
    def init(self):
        """初始化（预热离线模型，可能耗时数秒）。任何异常都不抛出。"""
        try:
            model_path, labels_path = _find_model_paths()
            if model_path is None:
                return {"ok": True, "model_loaded": False, "mode": self.mode,
                        "threshold": self.threshold, "classes": 0}
            try:
                import numpy as np  # noqa: F401  检查推理依赖
            except Exception:
                _log_offline_error("[init] ")
                return {"ok": True, "model_loaded": False, "mode": self.mode,
                        "threshold": self.threshold, "classes": 0}
            try:
                # 中文路径 bug：必须先读成字节再传 model_content=
                with open(model_path, "rb") as f:
                    model_content = f.read()
                interp = make_tflite_interpreter(model_content)
                interp.allocate_tensors()
                with open(labels_path, encoding="utf-8") as f:
                    labels = f.read().splitlines()
                self._interp = interp
                self._labels = labels
                self._loaded_model_path = model_path
                return {"ok": True, "model_loaded": True, "mode": self.mode,
                        "threshold": self.threshold, "classes": len(labels)}
            except Exception:
                _log_offline_error("[init] ")
                return {"ok": True, "model_loaded": False, "mode": self.mode,
                        "threshold": self.threshold, "classes": 0}
        except Exception as e:
            return {"ok": False, "error": str(e), "model_loaded": False,
                    "mode": self.mode, "threshold": self.threshold,
                    "classes": 0}

    def pick_and_recognize(self):
        """弹文件选择框并识别。识别流程与旧版完全一致。"""
        try:
            file_path = self._window.create_file_dialog(
                webview.OPEN_DIALOG, allow_multiple=False,
                file_types=('图片文件 (*.jpg;*.jpeg;*.png;*.bmp;*.gif)',
                            '所有文件 (*.*)'))
            if not file_path:
                return {"ok": False, "canceled": True}
            image_path = file_path if isinstance(file_path, str) else file_path[0]
        except Exception as e:
            return {"ok": False, "error": f"选择文件失败: {e}"}

        try:
            result = self._recognize(image_path)
            if not result.get("ok"):
                return result

            # 预览图：thumbnail 到 (400,300)，转 JPEG base64 data URL
            preview = ""
            try:
                img = Image.open(image_path)
                img.thumbnail((400, 300), Image.Resampling.LANCZOS)
                buf = io.BytesIO()
                img.convert("RGB").save(buf, format='JPEG', quality=85)
                preview = ("data:image/jpeg;base64,"
                           + base64.b64encode(buf.getvalue()).decode('utf-8'))
            except Exception:
                preview = ""

            return {"ok": True,
                    "mode_used": result["mode_used"],
                    "items": result["items"],
                    "top1": result["top1"],
                    "top1_score": result["top1_score"],
                    "preview": preview,
                    "path": image_path}
        except Exception as e:
            return {"ok": False, "error": f"识别失败: {str(e)}"}

    def ask(self, question):
        """AI 问答：调用 DeepSeek（OpenAI 兼容接口）"""
        try:
            question = (question or "").strip()
            if not question:
                return {"ok": False, "error": "问题不能为空"}

            # 构造 messages（system 含角色与可选的识别上下文）
            system_content = "你是中草药知识专家，回答简洁通俗。"
            if self.last_top1:
                system_content += (f"\n当前识别结果：{self.last_top1['name']}，"
                                   f"置信度{self.last_top1['score'] * 100:.2f}%。")
            messages = [{"role": "system", "content": system_content}]
            messages.extend(self.chat_messages)
            messages.append({"role": "user", "content": question})

            url = f"{_cfg('chat_api_base')}/chat/completions"
            headers = {
                "Authorization": f"Bearer {_cfg('chat_api_key')}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": _cfg("chat_model"),
                "messages": messages,
                "temperature": 0.7
            }
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            answer = data["choices"][0]["message"]["content"].strip()
            # 记录到历史（供多轮对话）
            self.chat_messages.append(messages[-1])          # user
            self.chat_messages.append({"role": "assistant",
                                       "content": answer})   # assistant
            if len(self.chat_messages) > 20:  # 只保留最近 10 轮
                self.chat_messages = self.chat_messages[-20:]
            return {"ok": True, "answer": answer}
        except Exception as e:
            return {"ok": False, "error": f"请求失败：{e}"}

    def set_mode(self, mode):
        try:
            if mode not in ("自动", "在线", "离线"):
                return {"ok": False, "error": f"无效模式: {mode}"}
            self.mode = mode
            self.save_mode()
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def set_threshold(self, v):
        try:
            v = float(v)
        except Exception:
            return {"ok": False, "error": f"无效阈值: {v}"}
        # 限幅 0.50~0.95
        if v < 0.50:
            v = 0.50
        if v > 0.95:
            v = 0.95
        self.threshold = v
        self.save_threshold(v)
        return {"ok": True}

    def open_link(self, url):
        try:
            webbrowser.open(url)
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ---------------- 识别链路（逻辑原样保留，与旧版一致） ----------------
    def _recognize(self, image_path):
        """三级识别链路：离线 → 通用物体兜底 → 在线。返回统一 dict。"""
        if not os.path.exists(image_path):
            return {"ok": False, "error": "图片文件不存在"}

        use_offline = (self.mode == "离线" or
                       (self.mode == "自动" and
                        os.path.exists(get_config_path("herb_model.tflite"))))
        if use_offline:
            offline = self.recognize_offline(image_path)
            if offline is not None:
                return self._format_result(offline, "离线")
            if self.mode == "离线":
                # 离线模式永不联网
                return {"ok": False,
                        "error": ("本地模型不可用（缺少 herb_model.tflite 或推理失败），\n"
                                  "请将模型文件放在程序同目录，或切换到\"在线\"模式")}

        access_token = self.get_access_token()
        if not access_token:
            return {"ok": False, "error": "获取百度 Token 失败"}

        with open(image_path, 'rb') as f:
            image_data = f.read()

        if len(image_data) > 4 * 1024 * 1024:
            image_data = self.compress_image(image_data)

        image_base64 = base64.b64encode(image_data).decode('utf-8')

        result = self.call_plant_api(access_token, image_base64)

        mode_used = "在线"
        if not result.get('result'):
            # 未匹配到中草药 → 通用物体识别兜底
            result = self.call_general_api(access_token, image_base64)
            result['_is_general'] = True
            mode_used = "在线(通用)"

        return self._format_result(result, mode_used)

    def _format_result(self, result, mode_used):
        """把百度/离线结果整理成统一 dict（含 top1 记录，供 AI 问答上下文）"""
        herbs = result.get('result') or []
        if not herbs:
            return {"ok": False,
                    "error": result.get('error_msg', "未识别到中草药")}

        items = []
        for herb in herbs:
            name = herb.get('name') or herb.get('keyword', '未知')
            score = herb.get('score', 0)
            desc = ""
            url = ""
            if 'baike_info' in herb:
                baike = herb['baike_info']
                description = baike.get('description', '')
                if description:
                    # 去换行（与旧版展示逻辑一致）
                    desc = description.replace('\n', ' ').strip()
                url = baike.get('baike_url', '') or ""
            items.append({"name": name, "score": float(score),
                          "desc": desc, "url": url})

        top = items[0]
        # 记录 top1 药材，供 AI 问答注入上下文
        self.last_top1 = {"name": top["name"], "score": top["score"]}
        return {"ok": True, "mode_used": mode_used, "items": items,
                "top1": top["name"], "top1_score": top["score"]}

    # ---------------- 离线推理（逻辑原样保留） ----------------
    def recognize_offline(self, image_path):
        model_path = labels_path = None
        if self._interp is None:                  # 未预热成功则现场加载
            model_path, labels_path = _find_model_paths()
            if model_path is None:
                return None

        try:
            import numpy as np

            if self._interp is not None:
                interp, labels = self._interp, self._labels
            else:
                # 中文路径 bug：必须先读成字节再传 model_content=
                with open(model_path, "rb") as f:
                    model_content = f.read()
                interp = make_tflite_interpreter(model_content)
                interp.allocate_tensors()
                with open(labels_path, encoding="utf-8") as f:
                    labels = f.read().splitlines()

            inp = interp.get_input_details()[0]
            out = interp.get_output_details()[0]

            img = Image.open(image_path).convert("RGB").resize((224, 224))
            # 输入喂原始 0-255 像素，不要手动归一化（模型第一层 Rescaling 已处理）
            interp.set_tensor(inp["index"], np.array(img, dtype=np.float32)[None])
            interp.invoke()
            scores = interp.get_tensor(out["index"])[0]
            top = scores.argsort()[-3:][::-1]

            return {"result": [{"name": labels[i], "score": float(scores[i])}
                               for i in top],
                    "_offline": True}
        except Exception:
            _log_offline_error()
            return None

    # ---------------- 百度 API（逻辑原样保留） ----------------
    def get_access_token(self):
        api_key = _cfg("baidu_api_key")
        secret_key = _cfg("baidu_secret_key")
        if not api_key or not secret_key:
            return None
        url = "https://aip.baidubce.com/oauth/2.0/token"
        params = {
            'grant_type': 'client_credentials',
            'client_id': api_key,
            'client_secret': secret_key
        }
        try:
            response = requests.post(url, params=params, timeout=10)
            result = response.json()
            if 'access_token' in result:
                return result['access_token']
            return None
        except Exception:
            return None

    def compress_image(self, image_data, max_size=(1024, 1024)):
        try:
            image = Image.open(io.BytesIO(image_data))
            image.thumbnail(max_size, Image.Resampling.LANCZOS)

            output = io.BytesIO()
            image.save(output, format='JPEG', quality=85)
            return output.getvalue()
        except Exception:
            return image_data

    def call_plant_api(self, access_token, image_base64):
        url = "https://aip.baidubce.com/rest/2.0/image-classify/v1/plant"
        headers = {'content-type': 'application/x-www-form-urlencoded'}
        data = {
            'access_token': access_token,
            'image': image_base64,
            'baike_num': 5
        }
        response = requests.post(url, data=data, headers=headers, timeout=30)
        return response.json()

    def call_general_api(self, access_token, image_base64):
        url = "https://aip.baidubce.com/rest/2.0/image-classify/v2/advanced_general"
        headers = {'content-type': 'application/x-www-form-urlencoded'}
        data = {
            'access_token': access_token,
            'image': image_base64,
            'baike_num': 5
        }
        response = requests.post(url, data=data, headers=headers, timeout=30)
        return response.json()


if __name__ == '__main__':
    api = HerbAPI()
    html = resource_path("index.html")   # frozen 时 sys._MEIPASS，否则脚本同目录
    window = webview.create_window(
        "智能中草药识别系统", html,
        js_api=api, width=1150, height=800, min_size=(980, 700))
    api._window = window
    try:
        webview.start(gui='edgechromium')
    except Exception:
        # edgechromium 不可用时回退默认 gui
        webview.start()
