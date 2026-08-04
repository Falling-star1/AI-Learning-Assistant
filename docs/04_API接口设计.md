# API 接口设计

## V1 调用方式

课程大作业第一版采用 Streamlit 直接调用 Python 模块，不强制拆出 HTTP 服务。核心调用关系如下：

```text
app.py -> modules.agent.router -> RAG / YOLO / LLM / Database
```

## 后续 FastAPI 预留接口

| 接口 | 方法 | 功能 |
| --- | --- | --- |
| `/api/chat` | POST | 用户提问，Agent 判断后返回回答 |
| `/api/knowledge/files` | POST | 上传课程资料并构建知识库 |
| `/api/knowledge/query` | POST | 对知识库进行检索问答 |
| `/api/detection` | POST | 上传图片并调用 YOLO 检测 |
| `/api/sessions/{session_id}` | GET | 获取指定会话历史 |
| `/api/reports` | POST | 根据对话和检测结果生成报告 |

## 通用返回格式

```json
{
  "success": true,
  "data": {},
  "message": "ok"
}
```

## 设计说明

- Streamlit 版本优先保证演示稳定性。
- Vue 升级时只需新增 FastAPI 层，复用 `modules/` 中的核心能力。
- 大模型、Embedding、YOLO 模型均通过配置项管理，便于本地和云端切换。