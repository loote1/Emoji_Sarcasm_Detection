# ESD-ZH 前端可视化展示

这个文件夹是为当前项目的 ESD-ZH emoji 讽刺检测模型制作的纯前端展示页。

## 使用真实模型启动

先安装 PyTorch：

```bash
python3 -m pip install -r model_visual_frontend/requirements.txt
```

然后在项目根目录运行：

```bash
python3 model_visual_frontend/server.py
```

浏览器打开：

```text
http://127.0.0.1:8080/
```

此时页面会调用 `POST /predict`，由 Python 后端加载 `artifacts/esd_zh/model.pt` 做真实推理。

## 接口格式

```http
POST /predict
Content-Type: application/json

{"text":"小心你活不到退休😄😄"}
```

返回示例：

```json
{
  "label": 1,
  "meaning": "讽刺 / 反话",
  "probability": 0.73,
  "tokens": ["T:小", "T:心", "T:你", "T:活", "T:不", "T:到", "T:退", "T:休", "E:😄", "E:😄"],
  "source": "real_model"
}
```

如果直接用 `file://` 打开 `index.html`，浏览器无法启动 PyTorch，因此会回退到前端模拟结果。
