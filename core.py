from pathlib import Path
import json, re
import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

def load_documents():
    return json.loads((DATA / "knowledge_base.json").read_text(encoding="utf-8"))

def load_graph():
    triples = json.loads((DATA / "knowledge_graph.json").read_text(encoding="utf-8"))
    graph = nx.MultiDiGraph()
    for t in triples:
        graph.add_edge(t["subject"], t["object"], relation=t["relation"])
    return graph, triples

def chunk_text(text, chunk_size=90, overlap=18):
    words = text.split()
    if not words:
        return []
    chunk_size = max(10, int(chunk_size))
    overlap = min(max(0, int(overlap)), chunk_size - 1)
    step = chunk_size - overlap
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), step)]

def prepare_chunks(documents):
    chunks = []
    for doc in documents:
        for idx, chunk in enumerate(chunk_text(doc["text"])):
            chunks.append({"text": chunk, "source": doc["source"], "topic": doc["topic"], "chunk_id": f'{doc["topic"]}_{idx}'})
    return chunks

class HybridAssistant:
    def __init__(self):
        self.documents = load_documents()
        self.chunks = prepare_chunks(self.documents)
        self.vectorizer = TfidfVectorizer(ngram_range=(1,2), strip_accents="unicode")
        self.matrix = self.vectorizer.fit_transform([c["text"] for c in self.chunks])
        self.graph, self.triples = load_graph()
        self.history = []

    def retrieve(self, query, top_k=3):
        q = self.vectorizer.transform([query])
        scores = cosine_similarity(q, self.matrix).ravel()
        order = np.argsort(scores)[::-1][:max(1, int(top_k))]
        results = []
        for i in order:
            if scores[i] > 0:
                item = dict(self.chunks[int(i)])
                item["score"] = float(scores[int(i)])
                results.append(item)
        return results

    def graph_facts(self, query):
        q = query.lower()
        terms = {n for n in self.graph.nodes if n.lower() in q}
        # Also match simple aliases.
        aliases = {"kg":"Knowledge Graph", "knowledge graph":"Knowledge Graph", "agent":"Intelligent Agent"}
        for alias, node in aliases.items():
            if alias in q and node in self.graph:
                terms.add(node)
        facts = []
        for node in terms:
            for _, target, data in self.graph.out_edges(node, data=True):
                facts.append(f"{node} --{data.get('relation','related_to')}--> {target}")
            for source, _, data in self.graph.in_edges(node, data=True):
                facts.append(f"{source} --{data.get('relation','related_to')}--> {node}")
        return list(dict.fromkeys(facts))[:12]

    def answer(self, query, top_k=3):
        query = query.strip()
        if not query:
            return {"answer":"Vui lòng nhập câu hỏi.", "sources":[], "facts":[], "tool":"none"}
        sources = self.retrieve(query, top_k)
        facts = self.graph_facts(query)
        self.history.append({"role":"user","content":query})
        if not sources and not facts:
            answer = ("Tôi chưa tìm thấy bằng chứng phù hợp trong kho tri thức hiện tại. "
                      "Hãy thử diễn đạt câu hỏi khác hoặc bổ sung tài liệu vào data/knowledge_base.json.")
        else:
            pieces = []
            if facts:
                pieces.append("Tri thức quan hệ:\n" + "\n".join(f"- {f}" for f in facts))
            if sources:
                pieces.append("Bằng chứng tài liệu:\n" + "\n".join(f"- [{s['topic']}; nguồn: {s['source']}] {s['text']}" for s in sources))
            answer = ("Dựa trên kho tri thức của hệ thống:\n\n" + "\n\n".join(pieces) +
                      "\n\nLưu ý: đây là bản triển khai minh họa dùng truy hồi TF-IDF và tổng hợp bằng quy tắc; "
                      "chưa gọi LLM sinh ngôn ngữ tự do. Hãy đối chiếu bằng chứng trước khi sử dụng.")
        self.history.append({"role":"assistant","content":answer})
        return {"answer":answer, "sources":sources, "facts":facts, "tool":"hybrid_retrieval"}

    def agent(self, query, top_k=3):
        # A transparent rule-based agent: selects tools and reports its plan.
        q = query.lower()
        plan = []
        if any(k in q for k in ["relation","relationship","graph","knowledge graph","quan hệ","đồ thị","kg"]):
            plan.append("Knowledge Graph query")
        if any(k in q for k in ["document","explain","what","why","how","rag","tài liệu","là gì","tại sao","như thế nào"]):
            plan.append("RAG document retrieval")
        if not plan:
            plan = ["RAG document retrieval", "Knowledge Graph query"]
        result = self.answer(query, top_k)
        result["agent_plan"] = plan
        result["agent_note"] = "Agent chọn công cụ theo luật minh bạch; chưa phải agent tự chủ dùng LLM."
        return result
