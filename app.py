import streamlit as st
from core import HybridAssistant

st.set_page_config(page_title="A7 Intelligent Assistant", page_icon="🧠", layout="wide")
st.title("A7 — Knowledge Graph + RAG + Intelligent Agent")
st.caption("Demo học thuật: truy hồi tài liệu, truy vấn đồ thị tri thức và agent điều phối công cụ.")
@st.cache_resource
def get_assistant():
    return HybridAssistant()
assistant = get_assistant()

with st.sidebar:
    st.header("Cấu hình")
    top_k = st.slider("Số đoạn tài liệu top-k", 1, 5, 3)
    mode = st.radio("Chế độ", ["Hybrid Assistant", "Intelligent Agent"])
    st.markdown("---")
    st.write("**Thành phần**")
    st.write("- TF-IDF semantic/lexical retrieval")
    st.write("- Knowledge Graph (NetworkX)")
    st.write("- Rule-based tool selection")
    st.write("- Evidence and source display")

if "messages" not in st.session_state:
    st.session_state.messages = []
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ví dụ: RAG hoạt động như thế nào?")
if prompt:
    st.session_state.messages.append({"role":"user","content":prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    result = assistant.agent(prompt, top_k) if mode == "Intelligent Agent" else assistant.answer(prompt, top_k)
    with st.chat_message("assistant"):
        st.markdown(result["answer"])
        if result.get("agent_plan"):
            st.markdown("**Kế hoạch agent:** " + " → ".join(result["agent_plan"]))
            st.caption(result["agent_note"])
        with st.expander("Bằng chứng được truy hồi", expanded=True):
            if result["facts"]:
                st.markdown("**Knowledge Graph facts**")
                for f in result["facts"]:
                    st.code(f)
            if result["sources"]:
                st.markdown("**Document chunks**")
                for s in result["sources"]:
                    st.markdown(f"**{s['topic']}** · `{s['source']}` · score `{s['score']:.3f}`")
                    st.write(s["text"])
            if not result["facts"] and not result["sources"]:
                st.info("Không có bằng chứng phù hợp.")
    st.session_state.messages.append({"role":"assistant","content":result["answer"]})

with st.expander("Kiểm tra Knowledge Graph"):
    st.write(f"Số node: {assistant.graph.number_of_nodes()} | Số cạnh: {assistant.graph.number_of_edges()}")
    st.dataframe([{"subject":t["subject"],"relation":t["relation"],"object":t["object"]} for t in assistant.triples], use_container_width=True)
