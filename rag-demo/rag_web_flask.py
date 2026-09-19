"""
RAG 知识库问答系统 · Flask Web版
轻量无依赖冲突，界面简洁实用
"""

import os
import json
import faiss
import numpy as np
from flask import Flask, render_template_string, request, jsonify
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================
# 配置
# ============================================================
MODEL_PATH = "/tmp/modelscope_cache/models/qwen--Qwen2.5-0.5B-Instruct/snapshots/master"
EMBED_MODEL_PATH = "/root/.cache/modelscope/models/damo--nlp_corom_sentence-embedding_chinese-base/snapshots/master"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
TOP_K_DEFAULT = 3

app = Flask(__name__)

# 全局状态
state = {
    "docs": [],
    "faiss_index": None,
    "embed_model": None,
    "llm_model": None,
    "tokenizer": None,
}


# ============================================================
# 工具函数
# ============================================================
def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """按字符切片，带重叠"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
        if start >= len(text):
            break
    return chunks


def load_embed_model():
    """加载embedding模型"""
    if state["embed_model"] is None:
        print("📥 加载 Embedding 模型...")
        state["embed_model"] = SentenceTransformer(EMBED_MODEL_PATH, trust_remote_code=True)
        print(f"✅ Embedding 加载完成")


def load_llm_model():
    """加载大模型"""
    if state["llm_model"] is None or state["tokenizer"] is None:
        print("📥 加载大模型...")
        state["tokenizer"] = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        state["llm_model"] = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, trust_remote_code=True, device_map="cpu"
        )
        print("✅ LLM: Qwen2.5-0.5B-Instruct")


def build_index(title, content):
    """从文本构建知识库索引"""
    chunks = split_text(content)
    if not chunks:
        return 0
    
    docs = []
    for i, chunk in enumerate(chunks):
        docs.append({
            "title": f"{title} · 片段{i+1}",
            "content": chunk
        })
    
    if state["embed_model"] is None:
        load_embed_model()
    
    # 向量化
    texts = [f"{d['title']}\n{d['content']}" for d in docs]
    embeddings = state["embed_model"].encode(texts)
    
    # 建FAISS索引
    state["faiss_index"] = faiss.IndexFlatL2(embeddings.shape[1])
    state["faiss_index"].add(embeddings.astype('float32'))
    state["docs"] = docs
    
    return len(docs)


def rag_query(question, top_k=TOP_K_DEFAULT, use_rag=True, temperature=0.3):
    """RAG问答"""
    if use_rag and state["faiss_index"] is not None and len(state["docs"]) > 0:
        if state["embed_model"] is None:
            load_embed_model()
        # 检索
        query_vec = state["embed_model"].encode([question]).astype('float32')
        distances, indices = state["faiss_index"].search(query_vec, k=top_k)
        
        # 拼接上下文
        context_parts = []
        sources = []
        for i, idx in enumerate(indices[0]):
            doc = state["docs"][idx]
            similarity = 1 / (1 + float(distances[0][i]))
            context_parts.append(f"【参考{i+1}：{doc['title']}】\n{doc['content']}")
            sources.append({
                "title": doc["title"],
                "score": round(similarity, 3),
                "snippet": doc["content"][:150] + "..."
            })
        
        context = "\n\n".join(context_parts)
        
        prompt = f"""你是一个专业的知识库助手。请根据以下参考资料回答用户的问题。
如果参考资料中找不到答案，请如实说"知识库中没有相关信息"，不要编造答案。
回答要简洁、准确、条理清晰。

【参考资料】
{context}

【用户问题】
{question}

【请回答】"""
    else:
        prompt = question
        sources = []
    
    # 生成回答
    if state["llm_model"] is None:
        load_llm_model()
    
    messages = [{"role": "user", "content": prompt}]
    text = state["tokenizer"].apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = state["tokenizer"](text, return_tensors="pt")
    
    outputs = state["llm_model"].generate(
        **inputs,
        max_new_tokens=500,
        temperature=temperature,
        do_sample=True,
        top_p=0.9,
        pad_token_id=state["tokenizer"].eos_token_id
    )
    answer = state["tokenizer"].decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    return answer, sources


# ============================================================
# 页面模板
# ============================================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG 知识库问答系统</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .header {
            text-align: center;
            color: white;
            margin-bottom: 30px;
        }
        .header h1 { font-size: 2em; margin-bottom: 8px; }
        .header p { opacity: 0.9; }
        .main {
            display: grid;
            grid-template-columns: 350px 1fr;
            gap: 20px;
            height: calc(100vh - 140px);
        }
        @media (max-width: 900px) {
            .main { grid-template-columns: 1fr; height: auto; }
        }
        .panel {
            background: white;
            border-radius: 16px;
            padding: 20px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            overflow-y: auto;
        }
        .panel h2 {
            font-size: 1.1em;
            color: #333;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #f0f0f0;
        }
        /* 左侧面板 */
        .upload-area {
            border: 2px dashed #ddd;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            margin-bottom: 15px;
            cursor: pointer;
            transition: all 0.3s;
        }
        .upload-area:hover { border-color: #667eea; background: #f8f9ff; }
        .upload-area.dragover { border-color: #667eea; background: #f0f4ff; }
        textarea {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 8px;
            font-family: inherit;
            font-size: 14px;
            resize: vertical;
            margin-bottom: 10px;
        }
        input[type="text"], input[type="file"] {
            width: 100%;
            padding: 8px 10px;
            border: 1px solid #ddd;
            border-radius: 6px;
            margin-bottom: 10px;
        }
        .btn {
            padding: 10px 20px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 600;
            transition: all 0.2s;
        }
        .btn-primary {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            width: 100%;
        }
        .btn-primary:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(102,126,234,0.4); }
        .btn-primary:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
        .status-badge {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
        }
        .status-ready { background: #d4edda; color: #155724; }
        .status-empty { background: #fff3cd; color: #856404; }
        .doc-list { margin-top: 10px; }
        .doc-item {
            padding: 8px 10px;
            background: #f8f9fa;
            border-radius: 6px;
            margin-bottom: 6px;
            font-size: 13px;
            color: #555;
        }
        .settings-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
        }
        .settings-row label { font-size: 13px; color: #555; }
        .settings-row input[type="range"] { width: 120px; }
        .toggle {
            position: relative;
            width: 44px;
            height: 24px;
            background: #ddd;
            border-radius: 12px;
            cursor: pointer;
            transition: background 0.3s;
        }
        .toggle.active { background: #667eea; }
        .toggle::after {
            content: '';
            position: absolute;
            top: 2px;
            left: 2px;
            width: 20px;
            height: 20px;
            background: white;
            border-radius: 50%;
            transition: transform 0.3s;
        }
        .toggle.active::after { transform: translateX(20px); }
        /* 右侧聊天 */
        .chat-container {
            display: flex;
            flex-direction: column;
            height: 100%;
        }
        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 10px 0;
        }
        .message {
            margin-bottom: 15px;
            display: flex;
        }
        .message.user { justify-content: flex-end; }
        .message-bubble {
            max-width: 75%;
            padding: 12px 16px;
            border-radius: 14px;
            line-height: 1.5;
            font-size: 14px;
        }
        .message.user .message-bubble {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border-bottom-right-radius: 4px;
        }
        .message.assistant .message-bubble {
            background: #f0f2f5;
            color: #333;
            border-bottom-left-radius: 4px;
        }
        .sources {
            margin-top: 10px;
            padding-top: 10px;
            border-top: 1px solid #ddd;
            font-size: 12px;
            color: #666;
        }
        .source-item {
            padding: 4px 0;
        }
        .source-score {
            display: inline-block;
            padding: 1px 6px;
            background: #e3f2fd;
            color: #1976d2;
            border-radius: 8px;
            font-size: 11px;
            margin-left: 6px;
        }
        .chat-input {
            display: flex;
            gap: 10px;
            padding-top: 15px;
            border-top: 1px solid #f0f0f0;
        }
        .chat-input input {
            flex: 1;
            padding: 12px 16px;
            border: 1px solid #ddd;
            border-radius: 12px;
            font-size: 14px;
            outline: none;
            transition: border 0.3s;
        }
        .chat-input input:focus { border-color: #667eea; }
        .chat-input button {
            padding: 12px 24px;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border: none;
            border-radius: 12px;
            cursor: pointer;
            font-weight: 600;
        }
        .chat-input button:disabled { opacity: 0.5; cursor: not-allowed; }
        .loading {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 2px solid #ddd;
            border-top-color: #667eea;
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        .empty-hint {
            text-align: center;
            color: #999;
            padding: 40px 20px;
        }
        .empty-hint h3 { color: #ccc; margin-bottom: 10px; font-size: 3em; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 RAG 知识库问答系统</h1>
            <p>检索增强生成 · 桥梁健康监测领域实战</p>
        </div>
        
        <div class="main">
            <!-- 左侧：知识库管理 -->
            <div class="panel">
                <h2>📚 知识库</h2>
                
                <div>
                    <label style="font-size:13px; color:#555;">文档标题</label>
                    <input type="text" id="docTitle" placeholder="例如：桥梁健康监测规范" value="桥梁健康监测知识库">
                </div>
                
                <div>
                    <label style="font-size:13px; color:#555;">粘贴文档内容</label>
                    <textarea id="docContent" rows="8" placeholder="在这里粘贴你的文档内容..."></textarea>
                </div>
                
                <button class="btn btn-primary" onclick="buildKB()" id="buildBtn">🚀 构建知识库</button>
                
                <div style="margin-top: 15px;">
                    <span style="font-size:13px; color:#555;">状态：</span>
                    <span class="status-badge status-empty" id="statusBadge">未构建</span>
                </div>
                
                <div class="doc-list" id="docList">
                    <div style="font-size:13px; color:#999; text-align:center; padding:20px 0;">
                        暂无文档
                    </div>
                </div>
                
                <h2 style="margin-top: 25px;">⚙️ 设置</h2>
                
                <div class="settings-row">
                    <label>启用 RAG 检索</label>
                    <div class="toggle active" id="ragToggle" onclick="toggleRAG()"></div>
                </div>
                
                <div class="settings-row">
                    <label>检索数量: <span id="topKValue">3</span></label>
                    <input type="range" min="1" max="10" value="3" id="topK" oninput="document.getElementById('topKValue').textContent=this.value">
                </div>
                
                <div class="settings-row">
                    <label>回答温度: <span id="tempValue">0.3</span></label>
                    <input type="range" min="0.1" max="1" step="0.1" value="0.3" id="temperature" oninput="document.getElementById('tempValue').textContent=this.value">
                </div>
            </div>
            
            <!-- 右侧：聊天 -->
            <div class="panel">
                <div class="chat-container">
                    <div class="chat-messages" id="chatMessages">
                        <div class="empty-hint" id="emptyHint">
                            <h3>💡</h3>
                            <p>先在左侧构建知识库</p>
                            <p>然后开始提问吧</p>
                        </div>
                    </div>
                    
                    <div class="chat-input">
                        <input type="text" id="userInput" placeholder="输入你的问题..." onkeypress="if(event.key==='Enter')sendMessage()">
                        <button onclick="sendMessage()" id="sendBtn">发送</button>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        let useRAG = true;
        
        function toggleRAG() {
            useRAG = !useRAG;
            document.getElementById('ragToggle').classList.toggle('active', useRAG);
        }
        
        function buildKB() {
            const title = document.getElementById('docTitle').value || '未命名文档';
            const content = document.getElementById('docContent').value;
            
            if (!content.trim()) {
                alert('请先粘贴文档内容');
                return;
            }
            
            const btn = document.getElementById('buildBtn');
            btn.disabled = true;
            btn.textContent = '⏳ 构建中...';
            
            fetch('/api/build', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ title, content })
            })
            .then(r => r.json())
            .then(data => {
                if (data.success) {
                    document.getElementById('statusBadge').textContent = data.count + ' 个片段';
                    document.getElementById('statusBadge').className = 'status-badge status-ready';
                    
                    const docList = document.getElementById('docList');
                    docList.innerHTML = data.docs.map((d, i) => 
                        `<div class="doc-item">📄 ${d.title}</div>`
                    ).join('');
                } else {
                    alert('构建失败: ' + data.error);
                }
            })
            .catch(e => alert('错误: ' + e))
            .finally(() => {
                btn.disabled = false;
                btn.textContent = '🚀 构建知识库';
            });
        }
        
        function sendMessage() {
            const input = document.getElementById('userInput');
            const question = input.value.trim();
            if (!question) return;
            
            // 清空提示
            const emptyHint = document.getElementById('emptyHint');
            if (emptyHint) emptyHint.remove();
            
            const chatDiv = document.getElementById('chatMessages');
            
            // 用户消息
            const userMsg = document.createElement('div');
            userMsg.className = 'message user';
            userMsg.innerHTML = `<div class="message-bubble">${escapeHtml(question)}</div>`;
            chatDiv.appendChild(userMsg);
            
            // AI 加载中
            const aiMsg = document.createElement('div');
            aiMsg.className = 'message assistant';
            aiMsg.innerHTML = `<div class="message-bubble"><span class="loading"></span> 正在思考...</div>`;
            chatDiv.appendChild(aiMsg);
            
            input.value = '';
            document.getElementById('sendBtn').disabled = true;
            
            // 滚动到底部
            chatDiv.scrollTop = chatDiv.scrollHeight;
            
            const topK = parseInt(document.getElementById('topK').value);
            const temperature = parseFloat(document.getElementById('temperature').value);
            
            fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question, top_k: topK, use_rag: useRAG, temperature })
            })
            .then(r => r.json())
            .then(data => {
                let html = escapeHtml(data.answer).replace(/\\n/g, '<br>');
                
                if (data.sources && data.sources.length > 0) {
                    html += '<div class="sources">📚 参考来源：';
                    data.sources.forEach((s, i) => {
                        html += `<div class="source-item">${i+1}. ${escapeHtml(s.title)} <span class="source-score">相似度 ${s.score}</span></div>`;
                    });
                    html += '</div>';
                }
                
                aiMsg.innerHTML = `<div class="message-bubble">${html}</div>`;
            })
            .catch(e => {
                aiMsg.innerHTML = `<div class="message-bubble" style="color:#e74c3c;">❌ 出错了: ${e}</div>`;
            })
            .finally(() => {
                document.getElementById('sendBtn').disabled = false;
                chatDiv.scrollTop = chatDiv.scrollHeight;
            });
        }
        
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
    </script>
</body>
</html>
"""


# ============================================================
# 路由
# ============================================================
@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/build", methods=["POST"])
def api_build():
    data = request.json
    title = data.get("title", "未命名文档")
    content = data.get("content", "")
    
    if not content.strip():
        return jsonify({"success": False, "error": "内容为空"})
    
    try:
        count = build_index(title, content)
        docs_preview = [{"title": d["title"]} for d in state["docs"][:10]]
        return jsonify({
            "success": True,
            "count": count,
            "docs": docs_preview
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.json
    question = data.get("question", "")
    top_k = data.get("top_k", TOP_K_DEFAULT)
    use_rag = data.get("use_rag", True)
    temperature = data.get("temperature", 0.3)
    
    try:
        answer, sources = rag_query(question, top_k=top_k, use_rag=use_rag, temperature=temperature)
        return jsonify({
            "success": True,
            "answer": answer,
            "sources": sources
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


# ============================================================
# 启动
# ============================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🔍 RAG 知识库问答系统 · Web版")
    print("=" * 60)
    
    print("\n🌐 启动 Web 服务 (端口 7860)...")
    print("打开浏览器访问: http://localhost:7860")
    print("模型将在首次使用时自动加载")
    print("=" * 60 + "\n")
    
    app.run(host="0.0.0.0", port=7860, debug=False, threaded=True)
