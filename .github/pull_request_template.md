## 改动说明

<!-- 简要描述这个 PR 做了什么，为什么做 -->

## 改动类型

<!-- 勾选适用的项 -->
- [ ] 新功能（feature）
- [ ] Bug 修复（fix）
- [ ] 文档更新（docs）
- [ ] 重构 / 优化（refactor）
- [ ] 依赖 / 配置（chore）

## 关联 Issue

<!-- 如 Fixes #12 / Refs #12，没有可留空 -->

## 自测情况

- [ ] 页面能正常启动：`streamlit run app.py`
- [ ] 测试全部通过：
  ```powershell
  $env:PYTHONPATH='.'; $env:LLM_PROVIDER='fallback'; python -m unittest discover -s tests -v
  ```
- [ ] README / 开发日志已同步更新
- [ ] 不包含 `.env`、API Key、数据库文件等敏感信息

## 截图 / 录屏

<!-- 如果是 UI 改动或新功能演示，请附截图 -->

## 备注

<!-- 其他需要 reviewer 注意的事项 -->
