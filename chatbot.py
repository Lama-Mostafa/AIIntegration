from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import pandas as pd

# Load and embed data (do this once)
df = pd.read_parquet("test-00000-of-00001-4e90740d45417ca4.parquet")
df["combined"] = df.apply(lambda row: f"{row['question']}\nChoices: {row['choices']}\nAnswer: {row['answer']}", axis=1)
chunks = [df["combined"][i] for i in range(len(df))]

# ChromaDB setup
embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="nutrition_chunks", embedding_function=embedding_fn)

# Only embed if collection is empty (avoid duplication)
if not collection.count():
    collection.add(documents=chunks, ids=[f"chunk_{i}" for i in range(len(chunks))])

# LangChain LLM
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
llm_model = OllamaLLM(
    model="deepseek-r1:14b",
    callback_manager=callback_manager,
    stop=["question:"],
    temperature=0.4,
    num_predict=-2,
    seed=1786
)

prompt_template = ChatPromptTemplate.from_template(
    "Context: {context}\nQuestion: {question}\n"
    "you are a nutritional chatbot for diabetic patients. After providing your response, "
    "suggest multiple meals and what to add (in grams) for each meal.\n"
    "If any user gives you a question in Arabic, answer in Arabic.\n"
    "Answer: Let's think step by step."
)

# RAG logic
def generate_response(question: str):
    print("thinking")
    results = collection.query(query_texts=[question], n_results=1, include=["documents", "distances"])

    context = results["documents"][0][0]
    distance = results["distances"][0][0]
    chain = prompt_template | llm_model
    answer = chain.invoke({"context": context, "question": question})
    return answer, context, distance
