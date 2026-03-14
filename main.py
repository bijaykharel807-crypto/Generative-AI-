import os
import json
import re
import networkx as nx
import streamlit as st
from groq import Groq
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize API clients
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ----------------------------------------------------------------------
# 1. Data and Knowledge Graph (same as before)
# ----------------------------------------------------------------------
RAW_DOCS = [
    {"id": 1, "text": "Metformin is first‑line for type 2 diabetes. It reduces hepatic glucose production."},
    {"id": 2, "text": "Insulin therapy may be added when HbA1c targets are not met with oral agents."},
    {"id": 3, "text": "SGLT2 inhibitors like empagliflozin reduce cardiovascular risk in diabetic patients."},
    {"id": 4, "text": "DPP‑4 inhibitors (e.g., sitagliptin) are weight‑neutral and have low hypoglycemia risk."},
    {"id": 5, "text": "GLP‑1 receptor agonists (e.g., liraglutide) promote weight loss and improve glycemic control."},
]

def clean_text(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()

def load_and_clean_docs():
    return [{"id": d["id"], "text": clean_text(d["text"])} for d in RAW_DOCS]

def extract_triples(text: str):
    prompt = f"""Extract medical knowledge triples (entity1, relation, entity2) from the text.
Return a JSON list of [entity1, relation, entity2]. Only output JSON.
Text: {text}"""
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        data = json.loads(content)
        triples = data.get("triples", [])
        return [tuple(t) for t in triples]
    except Exception as e:
        st.error(f"Triple extraction failed: {e}")
        return []

def build_knowledge_graph(docs):
    kg = nx.MultiDiGraph()
    for doc in docs:
        triples = extract_triples(doc["text"])
        for subj, rel, obj in triples:
            kg.add_edge(subj, obj, relation=rel, source=doc["id"])
    return kg

# ----------------------------------------------------------------------
# 2. Load or build KG (cached to avoid recomputation)
# ----------------------------------------------------------------------
@st.cache_resource
def get_knowledge_graph():
    if os.path.exists("medical_kg.gexf"):
        kg = nx.read_gexf("medical_kg.gexf")
        st.info("")
    else:
        with st.spinner(""):
            docs = load_and_clean_docs()
            kg = build_knowledge_graph(docs)
            nx.write_gexf(kg, "medical_kg.gexf")
            st.success("")
    return kg

# ----------------------------------------------------------------------
# 3. Retrieval and Answer Generation
# ----------------------------------------------------------------------
def retrieve_from_kg(question, kg, top_k=5):
    keywords = set(question.lower().split())
    relevant = []
    for u, v, data in kg.edges(data=True):
        if any(k in u.lower() or k in v.lower() for k in keywords):
            relevant.append(f"{u} {data['relation']} {v}")
    return relevant[:top_k]

def answer_with_kg(question, kg):
    context = retrieve_from_kg(question, kg)
    context_str = "\n".join(context) if context else "No specific facts retrieved."
    prompt = f"""You are a medical assistant. Use the following knowledge graph facts if relevant, otherwise rely on your own knowledge.
Knowledge Graph Facts:
{context_str}

Question: {question}
Answer:"""
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content, context
    except Exception as e:
        return f"Error: {e}", []

# ----------------------------------------------------------------------
# 4. Safety Moderation (optional)
# ----------------------------------------------------------------------
def is_safe(question):
    try:
        mod = openai_client.moderations.create(input=question)
        return not mod.results[0].flagged
    except:
        return True  # if moderation fails, allow

# ----------------------------------------------------------------------
# 5. Streamlit UI
# ----------------------------------------------------------------------
st.set_page_config(page_title="Medical QA Assistant", page_icon="💊")
st.title("💊 Medical QA Assistant ")

# Load knowledge graph
kg = get_knowledge_graph()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "context" in msg and msg["context"]:
            with st.expander("Retrieved knowledge graph facts"):
                for fact in msg["context"]:
                    st.write(f"- {fact}")

# Chat input
if prompt := st.chat_input("Ask a medical question..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Check safety
    if not is_safe(prompt):
        response = "I cannot answer that question due to safety policies."
        context = []
    else:
        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response, context = answer_with_kg(prompt, kg)
            st.markdown(response)
            if context:
                with st.expander("Retrieved knowledge graph facts"):
                    for fact in context:
                        st.write(f"- {fact}")

    # Save assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "context": context
    })