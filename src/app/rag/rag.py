from llama_index.core import ChatPromptTemplate,Document,Settings,SimpleDirectoryReader,VectorStoreIndex
from llama_index.llms.groq import Groq
from llama_index.llms.openai import OpenAI
from llama_index.llms.openai_like import OpenAILike
from llama_index.llms.ollama import Ollama


Settings.llm = Ollama(model="llama3.1")



documents = SimpleDirectoryReader(input_dir="./data",).load_data()
index = VectorStoreIndex.from_documents()
query_engine = index.as_query_engine()
response =  query_engine.query("what is documents about ")