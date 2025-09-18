# 法理通法律问答系统

基于大模型与法律知识库的智能法律问答平台，支持法律知识检索、智能问答、知识库管理、API服务与可视化 WebUI。

## 项目简介
本项目结合大语言模型与法律知识库，面向法律咨询、案例检索、法律知识管理等场景，提供高效、智能的法律问答服务。

## 主要功能
- **法律智能问答**：支持自然语言提问，结合大模型与知识库进行法律解答。
- **知识库检索与管理**：支持法律法规、案例等知识的存储、检索与管理。
- **API服务**：通过 FastAPI 提供 RESTful API，便于系统集成。
- **WebUI前端**：基于 Streamlit，支持在线问答与知识库管理。

## 安装方法
1. **环境准备**：建议使用 Python 3.10 及以上版本。
2. **依赖安装**：
   ```bash
   pip install -r requirements.txt
   ```

## 初始化与启动
1. **数据库初始化**（首次使用或知识库变更时）：
   ```bash
   python init_database.py --recreate-vs
   ```
   或仅初始化表结构：
   ```bash
   python init_database.py
   ```
2. **启动 API 服务**：
   ```bash
   python startup.py
   ```
3. **启动 WebUI 前端**：
   ```bash
   python webui.py
   ```
   或使用 Streamlit 命令：
   ```bash
   streamlit run webui.py
   ```

## 目录结构说明
- `chains/`：大模型推理链相关代码
- `common/`：通用工具与基础模块
- `configs/`：模型与服务配置
- `document_loaders/`：文档加载与解析
- `embeddings/`：向量化相关模块
- `knowledge_base/`：法律知识库及样例数据
- `server/`：后端服务与 API 实现
- `webui_pages/`：WebUI 页面与组件
- `tests/`：测试用例
- `docs/`：项目文档与说明

## 使用示例
- 访问 WebUI，输入法律问题进行智能问答。
- 通过 API 集成至其他系统，获取法律问答结果。
- 管理和扩展法律知识库，提升问答准确性。

## 贡献指南
欢迎提交 Issue、建议或 PR，详见 `CONTRIBUTING.md`。

## 许可证
本项目采用开源许可证，详见 `LICENSE` 文件。

---
如需详细安装与使用说明，请参考 `docs/` 目录下相关文档。

