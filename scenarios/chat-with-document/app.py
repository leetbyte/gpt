import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import Cohere
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA, ConversationChain
from langchain.memory import ConversationBufferWindowMemory, CombinedMemory, ConversationSummaryMemory

load_dotenv()

loader = PyPDFLoader(os.path.join(
    os.environ["SOURCE_DIR"], '.pdf'))
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings()

docsearch = Chroma.from_documents(
    texts,
    embeddings,
    persist_directory=os.path.join(os.environ["VECTOR_STORE_LOCAL_DIR"], ".chroma"))

llm = Cohere(cohere_api_key=os.environ["COHERE_API_KEY"],
             temperature=0, frequency_penalty=1, stop=["\n"])

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={"k": 1}))

# conv_memory = ConversationBufferWindowMemory(
#     memory_key="chat_history_lines",
#     input_key="input",
#     k=1
# )
# summary_memory = ConversationSummaryMemory(llm=llm, input_key="input")
# memory = CombinedMemory(memories=[conv_memory, summary_memory])

# default_template = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

# Summary of conversation:
# {history}
# Current conversation:
# {chat_history_lines}
# Human: {input}
# AI:"""

# prompt_template = PromptTemplate(
#     input_variables=["history", "input", "chat_history_lines"], template=default_template
# )
# conversation = ConversationChain(
#     llm=llm,
#     verbose=True,
#     memory=memory,
#     prompt=prompt_template
# )

query = "What certifications does he have?"
result = qa.run(query)

print(result)
