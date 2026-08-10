# AI 多模态学习助手

本项目是面向小学期课程大作业的 AI 多模态学习助手。系统采用 Streamlit + Python 模块化架构，支持课程资料上传、RAG 知识库问答、学习报告生成、YOLO 图片目标检测、Agent 自动路由、会话历史和模型 Provider 切换。

## 功能概览

- 课程资料问答：上传 PDF、TXT、Markdown、PPTX 后构建本地知识库，并基于检索片段回答问题。
- 引用来源展示：回答和报告会显示资料来源、片段编号和相关度，便于追溯依据。
- Agent 自动路由：根据用户输入和当前模式识别意图，自动调用 RAG、LLM、YOLO 或报告生成流程。
- 推理过程展示：自动路由回答下方可展开查看“识别意图 → 调用工具 → 返回结果”的执行过程。
- 图片目标检测：上传图片后调用 YOLOv8 进行检测，并可结合 LLM 生成自然语言分析。
- 学习辅助生成：生成课程总结、复习提纲、报告大纲或实验报告，并支持 Markdown 下载。
- 知识库与会话管理：支持多个知识库、多个会话、历史消息恢复和资料管理。
- 模型切换：支持云端兼容 API、本地 fallback、Ollama 本地模型，以及旧版 DashScope / DeepSeek 配置。

## 目录结构

```text
AI-Learning-Assistant/
├── app.py                    # Streamlit 主入口
├── config.py                 # 路径、模型和环境变量配置
├── requirements.txt          # Python 依赖
├── README.md
├── data/
│   ├── uploads/              # 上传课程资料
│   ├── images/               # 上传图片
│   ├── vector_db/            # 本地 RAG 存储
│   └── sqlite.db             # 运行后生成的 SQLite 数据库
├── modules/
│   ├── agent/                # Agent 路由和工具编排
│   ├── rag/                  # 文件解析、切分、检索、问答
│   ├── llm/                  # LLM Provider 和 Prompt
│   ├── yolo/                 # YOLO 检测与结果分析
│   ├── report/               # 报告生成
│   ├── database/             # SQLite 数据访问
│   └── utils/                # 文件和通用工具
├── outputs/
│   ├── detected_images/      # 检测结果图
│   ├── generated_reports/    # 生成报告输出目录
│   └── logs/
├── docs/                     # PRD、架构、数据库、API、任务拆分文档
└── tests/                    # 单元测试
```

## 环境要求

- **Python 3.11**（必须，其他版本可能存在兼容问题）
- 建议使用虚拟环境 `.venv` 隔离依赖
- 首次运行 YOLO 时会加载项目内置的 `yolov8n.pt` 权重，无需额外下载

## 安装步骤

1. 进入项目目录：

```bash
cd AI-Learning-Assistant
```

2. 创建并激活虚拟环境：

```bash
python -m venv .venv
.venv\Scripts\activate
```

3. 安装依赖（已锁定精确版本）：

```bash
pip install -r requirements.txt
```

4. 配置环境变量：

```bash
# 复制配置模板
Copy-Item .env.example .env
# 然后编辑 .env，填入你的 API Key 或选择本地模型
```

详细配置项说明见 `.env.example` 文件注释。

## 配置 LLM_API_KEY

项目启动时会自动读取 `AI-Learning-Assistant/.env`。如果需要调用云端大模型，在该目录下新建 `.env` 文件：

```env
LLM_PROVIDER=auto
LLM_API_KEY=你的 API Key
LLM_BASE_URL=https://你的兼容接口地址/v1
LLM_MODEL_NAME=你的模型名称
```

说明：

- `LLM_PROVIDER=auto`：优先使用 `LLM_API_KEY` 配置的云端兼容 API；没有 Key 时自动进入本地 fallback。
- `LLM_PROVIDER=cloud`：只尝试使用云端兼容 API；没有 `LLM_API_KEY` 时 fallback。
- `LLM_PROVIDER=fallback`：不调用外部大模型；简单闲聊和算术会给出本地友好回复，RAG 问答会返回检索到的课程片段，适合无网络答辩演示。
- `LLM_PROVIDER=ollama`：显式调用本地 Ollama，默认地址为 `http://localhost:11434`，默认模型为 `qwen2.5:7b`。
- 兼容旧配置：也可以使用 `DASHSCOPE_API_KEY` 或 `DEEPSEEK_API_KEY`。

如果使用 OpenAI 兼容服务，通常需要同时配置 `LLM_API_KEY`、`LLM_BASE_URL` 和 `LLM_MODEL_NAME`。如果服务商使用默认 OpenAI 地址，可以不填 `LLM_BASE_URL`。

## 运行应用

```bash
streamlit run app.py
```

启动后浏览器会打开本地页面，默认地址通常是：

```text
http://localhost:8501
```

推荐演示流程：

1. 在侧边栏新建或选择知识库。
2. 上传课程 PDF / TXT / Markdown / PPTX，等待索引完成。
3. 选择“自动识别”或“课程资料问答”，输入课程相关问题。
4. 展开回答下方的“查看推理过程”和“查看引用来源”。
5. 切换到“学习辅助生成”，生成课程总结或实验报告并下载 Markdown。
6. 切换到“图片目标检测”，上传图片并查看 YOLO 检测结果。

## 运行测试

测试入口会跳过 `.env` 加载，并在测试包中固定 provider 环境，避免因本机配置了 API Key 或 Ollama 导致测试触网失败：

```powershell
cd AI-Learning-Assistant
$env:PYTHONPATH='.'
python -m unittest discover -s tests -v
```

运行单个测试文件：

```powershell
$env:PYTHONPATH='.'; python -m unittest tests.test_route_knowledge_base -v
```

## 交付验收用例

建议答辩或交付前在默认“自动识别”模式下按顺序验证：

| 输入 | 期望结果 |
|---|---|
| `你好` | 走普通问答，返回友好回复，不出现“未检索到相关片段”。 |
| `你是谁` | 走普通问答，说明助手能力，不误走 RAG。 |
| `1+1等于几` | 走普通问答，返回基础计算结果。 |
| `解释 CPU 指令周期` | 走课程资料问答，回答带引用来源。 |
| `解释量子纠缠` | 若课程资料未覆盖，走 LLM 通用回答，并标注“未在课程资料中检索到相关内容，以下为通用回答”。 |

交付前命令：

```powershell
python -m unittest discover tests
streamlit run app.py --server.port 8501
```

## 常见问题

**没有配置 API Key 能不能演示？**
可以。系统会进入 fallback 模式，RAG 问答会返回检索到的课程片段，适合展示资料解析、检索、引用来源和 Agent 工作流。

**PDF 上传后提示解析失败怎么办？**
请确认 PDF 没有损坏，并且不是纯扫描版。当前版本主要读取可复制文本的 PDF；扫描版 PDF 需要先 OCR。

**PPT 文件为什么不能上传？**
当前支持 `.pptx`，不支持旧版 `.ppt`。请先转换为 `.pptx`。

**YOLO 首次运行很慢怎么办？**
首次运行可能会下载模型权重并初始化模型，后续会更快。答辩前建议提前运行一次图片检测流程。

**如何切换本地模型？**
先启动 Ollama 并拉取模型，例如 `ollama pull qwen2.5:7b`，然后在 `.env` 中设置 `LLM_PROVIDER=ollama`，或在页面侧边栏选择“本地模型”。

## 团队协作

本项目使用 GitHub 进行协同开发，采用 **main + feature 分支 + Pull Request** 工作流。

### 分支命名规范

```
feature/<功能名>     # 新功能，如 feature/rag-splitter-tune
fix/<问题名>         # 修 bug，如 fix/provider-network
docs/<文档名>        # 文档更新，如 docs/final-report
chore/<杂项>         # 依赖升级、配置调整等
```

### 分工建议

| 角色 | 负责模块 | 主要文件 |
|---|---|---|
| A 前端/演示 | Streamlit 页面、UI、工作流展示 | `app.py`、`README.md` |
| B RAG/资料问答 | 文件解析、检索、引用来源、知识库 | `modules/rag/`、`modules/database/` |
| C Agent/LLM/YOLO | Agent 路由、模型切换、YOLO 检测 | `modules/agent/`、`modules/llm/`、`modules/yolo/` |
| D 文档/答辩 | PRD、架构图、开发日志、答辩 PPT | `docs/`、`README.md` |

> `app.py` 是冲突高发区，建议由 A 统一接入，B/C 改完模块后通知 A 对接 UI。

### 日常开发流程

```powershell
# 1. 开始工作前同步最新代码
git checkout main
git pull origin main

# 2. 切功能分支
git checkout -b feature/your-feature

# 3. 改代码、本地测试
$env:PYTHONPATH='.'; $env:LLM_PROVIDER='fallback'; python -m unittest discover -s tests -v

# 4. 提交（提交信息按规范填写）
git add <具体文件>
git commit -m "feat(rag): 优化切分策略，支持按段落切分"

# 5. 推送并开 PR
git push -u origin feature/your-feature
# 然后去 GitHub 网页发起 Pull Request，@组员 review
```

### 提交信息规范

```
feat(模块): 简短描述      # 新功能
fix(模块): 简短描述       # 修 bug
docs: 简短描述            # 文档更新
chore: 简短描述           # 依赖、配置等杂项
```

模块可填：`rag` / `llm` / `yolo` / `agent` / `report` / `ui` / `db` 等。

### PR 合并前检查清单

- [ ] 页面能正常启动：`streamlit run app.py`
- [ ] 测试全部通过：`python -m unittest discover -s tests -v`
- [ ] README / 开发日志已同步更新
- [ ] 至少一名组员 review 通过

### 协作注意事项

- **不要多人同时大改 `app.py`**，很容易冲突
- **每个 PR 只做一件事**，不要把多个功能混在一个 PR
- **`.env` 只在本地放**，不要提交到仓库（已被 `.gitignore` 忽略）
- **小步快跑**，频繁提交，减少冲突
- **冲突优先沟通**，不要盲目 `git checkout --theirs/--ours`

## 开发原则

Streamlit 只负责页面展示和用户交互；RAG、LLM、YOLO、Agent、数据库等能力封装在 `modules/` 中，便于后续升级为 Vue + FastAPI 架构。
