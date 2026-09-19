"""
RAG 检索增强生成 · 桥梁健康知识库实战
===========================================
RAG = Retrieval-Augmented Generation（检索增强生成）

原理：用户提问 → 从知识库检索相关片段 → 把问题+相关片段一起给模型 → 模型基于资料回答
优点：减少幻觉、回答有依据、知识可以随时更新
"""

import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================
# 配置
# ============================================================
MODEL_PATH = "/tmp/modelscope_cache/models/qwen--Qwen2.5-0.5B-Instruct/snapshots/master"
EMBED_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"  # 多语言小embedding模型
OUTPUT_DIR = "/Coze/Drive/扣子/所有对话/主对话/bridge-ai-train/rag_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 第一步：准备知识库（桥梁健康领域文档）
# ============================================================
def build_knowledge_base():
    """构建桥梁健康知识库（实际项目中从Word/PDF/数据库读取）"""
    print("=" * 60)
    print("📚 Step 1: 构建桥梁健康知识库")
    print("=" * 60)
    
    # 模拟从各种文档中提取的知识片段
    # 实际项目中：用文档解析工具从PDF/Word/Excel中提取，再做切片
    knowledge_docs = [
        {
            "id": 1,
            "title": "桥梁健康监测系统技术规范",
            "content": "桥梁健康监测系统（Bridge Health Monitoring System, BHMS）是通过安装在桥梁上的各类传感器，"
                      "对桥梁结构响应、环境参数和荷载作用进行长期、连续、实时监测，"
                      "并通过数据分析与评估，掌握桥梁结构的技术状态和安全性能的系统。"
                      "完整的监测系统通常包括：传感器子系统、数据采集与传输子系统、数据处理与分析子系统、"
                      "预警与决策子系统、可视化展示子系统五个部分。"
        },
        {
            "id": 2,
            "title": "传感器布设原则",
            "content": "桥梁健康监测传感器布设应遵循以下原则：1）目的性原则：测点布设必须紧密围绕监测目标，"
                      "避免盲目追求大而全；2）代表性原则：测点应布置在结构响应最大、最能反映结构状态的位置；"
                      "3）可靠性原则：传感器安装位置应便于施工、维护和更换，尽量减少人为损坏和环境干扰；"
                      "4）经济性原则：在满足监测需求的前提下，合理控制测点数量和成本；"
                      "5）冗余性原则：关键部位和重要参数应有一定的冗余设计，提高系统可靠性。"
        },
        {
            "id": 3,
            "title": "桥梁预警等级划分标准",
            "content": "根据《桥梁结构健康监测系统设计规范》，桥梁监测预警等级一般划分为四级："
                      "1）蓝色预警（四级/提示）：监测指标略有波动但仍在正常范围内，需加强关注；"
                      "2）黄色预警（三级/注意）：监测指标超过正常值但未超出设计限值，建议加密观测并安排检查；"
                      "3）橙色预警（二级/警示）：监测指标接近或局部超过设计限值，结构可能存在隐患，需专项检测评估；"
                      "4）红色预警（一级/警告）：监测指标严重超限，结构安全可能受威胁，应立即限制交通并启动应急预案。"
                      "预警阈值的确定应结合设计标准、历史统计数据和专家经验综合确定。"
        },
        {
            "id": 4,
            "title": "光纤光栅传感器工作原理",
            "content": "光纤布拉格光栅（Fiber Bragg Grating, FBG）传感器是一种基于光纤光栅的波长调制型传感器。"
                      "其工作原理是：在光纤纤芯内写入周期性折射率调制的光栅结构，当宽带光入射时，"
                      "满足布拉格条件的特定波长的光被反射，反射光的中心波长称为布拉格波长。"
                      "当光纤受到轴向应变或温度变化时，光栅周期和纤芯折射率发生变化，"
                      "导致布拉格波长漂移，通过精确测量波长漂移量即可计算出应变或温度的变化值。"
                      "FBG传感器具有抗电磁干扰、耐腐蚀、精度高、可串联复用等优点。"
        },
        {
            "id": 5,
            "title": "温度对桥梁位移的影响",
            "content": "温度是影响桥梁位移的最主要环境因素之一。桥梁结构在温度变化下产生热胀冷缩变形，"
                      "导致桥梁各部位产生位移和变形。对于大跨度桥梁，温度引起的位移往往占总位移的60%~90%。"
                      "温度对桥梁位移的影响具有以下特点：1）周期性：随气温日变化和季节变化呈周期性波动；"
                      "2）滞后性：结构内部温度变化滞后于气温变化，大体积混凝土尤为明显；"
                      "3）分布不均：桥梁不同部位、不同深度的温度分布不均匀，产生温度梯度效应；"
                      "4）可分离性：温度效应与结构损伤引起的位移变化具有不同的特征，可以通过数据方法分离。"
                      "在桥梁健康评估中，通常需要先剔除温度效应再分析结构本身的状态变化。"
        },
        {
            "id": 6,
            "title": "超限车辆对桥梁的危害",
            "content": "超限车辆对桥梁结构安全造成严重危害，主要表现在以下几个方面："
                      "1）结构承载能力下降：超载使桥梁构件应力超过设计值，加速材料疲劳和损伤积累；"
                      "2）冲击效应加剧：重型车辆行驶产生的冲击振动比设计荷载大得多，加剧结构疲劳损伤；"
                      "3）桥面铺装损坏：超载导致桥面铺装出现车辙、开裂、拥包等病害；"
                      "4）支座和伸缩缝加速老化：超重荷载使支座压应力超限，加速橡胶老化和钢板变形；"
                      "5）下部结构病害：桥墩基础承受过大荷载，可能产生不均匀沉降和滑移；"
                      "6）安全储备降低：超载使桥梁安全系数大幅降低，极端情况下可能引发垮桥事故。"
                      "研究表明，车辆超限30%，桥梁疲劳寿命缩短约60%。"
        },
        {
            "id": 7,
            "title": "桥梁定期检测频率要求",
            "content": "根据《公路桥涵养护规范》JTG 5120-2021，桥梁定期检查频率规定如下："
                      "1）一类桥梁（技术状况评定为1类）：每3年检查一次；"
                      "2）二类桥梁（技术状况评定为2类）：每2年检查一次；"
                      "3）三类桥梁（技术状况评定为3类）：每年检查一次；"
                      "4）四类桥梁（技术状况评定为4类）：每半年检查一次，并进行交通管制；"
                      "5）五类桥梁（技术状况评定为5类）：立即进行交通封闭，安排加固或拆除重建。"
                      "特殊结构桥梁（特大跨径桥梁、新型结构桥梁）应适当提高检查频率。"
                      "健康监测系统可以作为定期检查的重要补充，但不能替代人工检查。"
        },
        {
            "id": 8,
            "title": "模态分析在桥梁监测中的应用",
            "content": "模态分析是桥梁健康监测中重要的结构状态评估方法。结构模态参数包括固有频率、阻尼比和振型。"
                      "当结构出现损伤时，局部刚度下降，会导致固有频率降低、振型发生变化。"
                      "模态分析在桥梁监测中的主要应用包括："
                      "1）损伤识别：通过频率变化率、模态曲率变化、柔度矩阵变化等识别损伤位置和程度；"
                      "2）模型校准：用实测模态参数修正有限元模型，提高分析精度；"
                      "3）状态评估：将监测模态参数与基准状态对比，评估结构整体健康状况；"
                      "4）荷载识别：结合模态参数和响应数据，反演作用在桥上的荷载。"
                      "需要注意的是，温度、湿度等环境因素也会影响模态参数，分析时需要进行修正。"
        },
        {
            "id": 9,
            "title": "桥梁健康监测数据预处理方法",
            "content": "桥梁健康监测数据预处理是数据分析的基础和前提。常用的预处理方法包括："
                      "1）异常值识别与处理：采用3σ准则、格拉布斯检验等方法识别异常值，根据情况进行修正或删除；"
                      "2）缺失值填补：对于短时数据缺失采用线性插值、样条插值等方法填补；"
                      "长时缺失需要标记并单独处理；"
                      "3）降噪处理：常用方法包括滑动平均、小波去噪、经验模态分解（EMD）等；"
                      "4）趋势项分离：将温度等环境因素引起的趋势项从数据中分离，"
                      "常用方法有回归分析、卡尔曼滤波、盲源分离等；"
                      "5）数据归一化：将不同量纲的数据归一化到统一尺度，便于对比分析；"
                      "6）时间同步：确保不同传感器数据的时间戳精确对齐。"
        },
        {
            "id": 10,
            "title": "中小桥梁监测方案设计要点",
            "content": "中小跨径桥梁数量众多，监测方案设计需要兼顾效果和成本。设计要点包括："
                      "1）分级监测：根据桥梁重要性、技术状况和风险等级，将桥梁分为不同监测等级，"
                      "配置不同的监测内容和精度；2）精简高效：重点监测关键部位和控制断面，"
                      "不必追求面面俱到，一般中小桥监测5~15个测点即可；"
                      "3）低成本方案：优先选用性价比高的传感器和传输方式，"
                      "如MEMS传感器、无线传输、太阳能供电等；4）集群化管理：建立区域级桥梁监测平台，"
                      "统一管理区域内数十甚至上百座桥梁，降低单桥成本；"
                      "5）适度智能：利用AI算法自动异常检测和预警，减少人工巡检工作量。"
        },
    ]
    
    # 保存知识库
    kb_path = os.path.join(OUTPUT_DIR, "knowledge_base.json")
    with open(kb_path, "w", encoding="utf-8") as f:
        json.dump(knowledge_docs, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 知识库构建完成，共 {len(knowledge_docs)} 篇文档")
    for doc in knowledge_docs:
        print(f"   [{doc['id']}] {doc['title']} ({len(doc['content'])}字)")
    print(f"   已保存到: {kb_path}")
    print()
    return knowledge_docs


# ============================================================
# 第二步：文档向量化（Embedding）
# ============================================================
def build_vector_index(docs):
    """将文档转换为向量，建立FAISS索引"""
    print("=" * 60)
    print("🔢 Step 2: 文档向量化 + 建立向量索引")
    print("=" * 60)
    
    # 加载embedding模型（多语言小模型，速度快）
    print("加载Embedding模型...")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    
    # 将每篇文档转成向量
    print("计算文档向量...")
    texts = [f"{doc['title']}\n{doc['content']}" for doc in docs]
    embeddings = embed_model.encode(texts, show_progress_bar=True)
    
    print(f"\n✅ 向量化完成")
    print(f"   文档数量: {len(embeddings)}")
    print(f"   向量维度: {embeddings.shape[1]}")
    
    # 建立FAISS索引
    index = faiss.IndexFlatL2(embeddings.shape[1])  # L2距离
    index.add(embeddings.astype('float32'))
    
    # 保存索引和元数据
    faiss.write_index(index, os.path.join(OUTPUT_DIR, "faiss_index.bin"))
    
    print(f"   FAISS索引已建立，索引向量数: {index.ntotal}")
    print()
    
    return embed_model, index


# ============================================================
# 第三步：检索演示
# ============================================================
def demo_retrieval(embed_model, index, docs):
    """演示检索效果"""
    print("=" * 60)
    print("🔍 Step 3: 检索效果演示")
    print("=" * 60)
    
    test_queries = [
        "桥梁预警分几级？分别是什么？",
        "超限车辆对桥有什么危害？",
        "光纤光栅传感器原理是什么？",
    ]
    
    for query in test_queries:
        print(f"\n❓ 问题: {query}")
        
        # 把问题转成向量
        query_vec = embed_model.encode([query]).astype('float32')
        
        # 检索最相似的3篇文档
        distances, indices = index.search(query_vec, k=3)
        
        print(f"   检索到的Top3文档:")
        for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            doc = docs[idx]
            # 相似度分数：距离越小越相似，转成0~1的分数
            similarity = 1 / (1 + dist)
            print(f"   {i+1}. [{similarity:.3f}] {doc['title']}")
    
    print("\n")
    return test_queries


# ============================================================
# 第四步：RAG 问答（检索 + 生成）
# ============================================================
def rag_qa(embed_model, index, docs, llm_model, tokenizer, query, top_k=3):
    """RAG问答：先检索，再基于检索结果生成回答"""
    
    # 1. 检索相关文档
    query_vec = embed_model.encode([query]).astype('float32')
    distances, indices = index.search(query_vec, k=top_k)
    
    # 2. 拼接检索到的文档作为上下文
    context_parts = []
    for i, idx in enumerate(indices[0]):
        doc = docs[idx]
        context_parts.append(f"【文档{i+1}：{doc['title']}】\n{doc['content']}")
    
    context = "\n\n".join(context_parts)
    
    # 3. 构造prompt，让模型基于上下文回答
    prompt = f"""你是桥梁健康监测领域的专业助手。请根据以下参考资料回答用户的问题。
如果参考资料中找不到答案，请如实说"参考资料中没有相关信息"，不要编造答案。
回答要简洁、准确、专业。

【参考资料】
{context}

【用户问题】
{query}

【请回答】"""
    
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt")
    
    outputs = llm_model.generate(
        **inputs,
        max_new_tokens=300,
        temperature=0.3,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    # 返回回答和检索到的来源
    retrieved_docs = [docs[idx] for idx in indices[0]]
    return response, retrieved_docs


# ============================================================
# 第五步：对比：有 RAG vs 无 RAG
# ============================================================
def compare_rag_vs_baseline(embed_model, index, docs, llm_model, tokenizer):
    """对比有RAG和无RAG的回答差异"""
    print("=" * 60)
    print("⚖️  Step 4: RAG vs 原始模型 对比")
    print("=" * 60)
    
    test_questions = [
        "桥梁预警分几级？分别是什么？",
        "超限车辆对桥梁有什么危害？",
        "中小桥梁监测方案怎么设计？",
    ]
    
    for query in test_questions:
        print(f"\n{'='*60}")
        print(f"❓ 问题: {query}")
        print(f"{'='*60}")
        
        # --- 无RAG（直接问模型）---
        print(f"\n📝 [无RAG] 模型直接回答：")
        print("-" * 40)
        messages = [{"role": "user", "content": query}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt")
        outputs = llm_model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        baseline_answer = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        print(baseline_answer)
        
        # --- 有RAG（检索+生成）---
        print(f"\n📝 [有RAG] 检索后回答：")
        print("-" * 40)
        rag_answer, retrieved = rag_qa(embed_model, index, docs, llm_model, tokenizer, query)
        print(rag_answer)
        
        print(f"\n📚 参考来源:")
        for i, doc in enumerate(retrieved):
            print(f"   {i+1}. {doc['title']}")
    
    print(f"\n{'='*60}")
    print("✅ RAG 对比完成！")
    print(f"\n💡 观察要点：")
    print("1. 无RAG时模型可能泛泛而谈，或给出不准确的信息")
    print("2. 有RAG时回答更具体、更专业，因为基于你的知识库")
    print("3. RAG回答可以追溯到具体文档来源，增加可信度")
    print(f"{'='*60}")


# ============================================================
# 主函数
# ============================================================
def main():
    print("\n" + "=" * 60)
    print("🔍 RAG 检索增强生成 · 桥梁健康知识库实战")
    print("=" * 60)
    print(f"大模型: Qwen2.5-0.5B-Instruct")
    print(f"Embedding: {EMBED_MODEL_NAME}")
    print(f"向量库: FAISS (本地文件)")
    print()
    
    # Step 1: 构建知识库
    docs = build_knowledge_base()
    
    # Step 2: 向量化 + 建索引
    embed_model, index = build_vector_index(docs)
    
    # Step 3: 检索演示
    demo_retrieval(embed_model, index, docs)
    
    # 加载大模型
    print("=" * 60)
    print("🤖 加载大模型用于RAG问答生成...")
    print("=" * 60)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    llm_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True, device_map="cpu"
    )
    print("✅ 大模型加载完成\n")
    
    # Step 4: RAG vs 无RAG对比
    compare_rag_vs_baseline(embed_model, index, docs, llm_model, tokenizer)


if __name__ == "__main__":
    main()
