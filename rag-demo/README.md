# RAG 桥梁健康知识库问答系统

基于 Qwen2.5-0.5B + FAISS 向量检索 + Sentence-Embedding 的 RAG 实战系统。

## 文件说明

- `rag_web_flask.py` - Flask Web 版 RAG 系统（带界面，推荐）
- `rag_bridge_knowledge.py` - 命令行版 RAG 演示（含对比实验）

## 快速开始

### 1. 安装依赖

```bash
pip install torch transformers sentence-transformers faiss-cpu flask modelscope
```

### 2. 下载模型

```python
# 从 ModelScope 下载（国内速度快）
from modelscope import snapshot_download

# 大模型
snapshot_download('qwen/Qwen2.5-0.5B-Instruct')

# Embedding 模型
snapshot_download('damo/nlp_corom_sentence-embedding_chinese-base')
```

### 3. 修改模型路径

在 `rag_web_flask.py` 中修改 `MODEL_PATH` 和 `EMBED_MODEL_PATH` 为你的本地模型路径。

### 4. 启动 Web 服务

```bash
python3 rag_web_flask.py
```

浏览器打开 http://localhost:7860

## 功能特性

- 📚 文档上传 + 自动切片 + 向量索引构建
- 🔍 语义相似度检索（Top-K 可调）
- 💬 智能问答 + 参考来源展示
- ⚙️ RAG 开关对比、温度调节
- 🎨 简洁美观的 Web 界面

## 技术栈

| 组件 | 技术 |
|------|------|
| 大模型 | Qwen2.5-0.5B-Instruct |
| 向量化 | 达摩院中文 Embedding |
| 向量库 | FAISS (Facebook) |
| Web框架 | Flask |

## 进阶优化建议

1. **混合检索**：BM25 + 向量检索合并
2. **Rerank**：加一层重排序模型
3. **父子切片**：小段检索，大段生成
4. **查询改写**：大模型优化用户问题后再检索
5. **评估体系**：建立召回率/准确率/回答质量三维评估
