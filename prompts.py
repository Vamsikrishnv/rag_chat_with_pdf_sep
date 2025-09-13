from langchain.prompts import ChatPromptTemplate
def get_rag_prompt():
    system = ("You are a precise research assistant. "
              "Answer using only the provided context. "
              "If unknown, say you don't know. Cite as (p. <page>).")
    human = "Question:\n{question}\n\nContext:\n{context}\n\nReturn a concise answer with citations."
    return ChatPromptTemplate.from_messages([('system', system), ('human', human)])
