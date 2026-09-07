from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
import time

llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", thinking_budget=0)

def call_model(state: MessagesState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

graph = StateGraph(MessagesState)
graph.add_node(call_model)
graph.add_edge(START, "call_model")
graph.add_edge("call_model", END)
graph = graph.compile()

print("calling model...", flush=True)
t0 = time.time()
result = graph.invoke({"messages": [{"role": "user", "content": "hi!"}]})
print(f"got response in {time.time()-t0:.1f}s", flush=True)

content = result["messages"][-1].content
if isinstance(content, list):
    text = content[0]["text"]
else:
    text = content
print(text)