import ollama
import pandas as pd
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEndpoint

class RAGEngine:
    def __init__(self, doc_dir: Path):
        self.doc_dir = doc_dir
        self.db = self._initialize_db()

    def _initialize_db(self):
        loader = DirectoryLoader(self.doc_dir, glob="./*.pdf", loader_cls=PyPDFLoader)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=100)
        texts = text_splitter.split_documents(docs)
        embeddings = HuggingFaceEmbeddings(model="nomic-ai/nomic-embed-text-v1.5")
        return Chroma.from_documents(documents=texts, embedding=embeddings, collection_name="rag")

    def retrieve(self, query: str, k: int = 4):
        query_result = self.db.similarity_search(query, k=4)
        return "\n".join(doc.page_content for doc in query_result)

def read_csv(data_path: Path):
    df = pd.read_csv(data_path)
    essay_id = df.get('essay_id')
    essay_content = df.get('full_text')
    essay_score = df.get('score')
    return essay_id, essay_content, essay_score

REFERENCE_TOOL = {
    'type': 'function',
    'function': {
        'name': 'retrieve_reference_info',
        'description': 'Retrieve scoring rubrics and reference criteria for essay scoring.',
        'parameters': {
            'type': 'object',
            'properties': {
                'query': {
                    'type': 'string',
                    'description': 'The search query to find relevant scoring criteria.',
                },
            },
            'required': ['query'],
        },
    },
}

def handle_tool_calls(tool_calls, rag_engine):
    available_functions = {
        'retrieve_reference_info': rag_engine.retrieve,
    }
    tool_responses = []
    for tool_call in tool_calls:
        function_name = tool_call['function']['name']
        function_args = tool_call['function']['arguments']
        if function_name in available_functions:
            print(f"Executing tool: {function_name} with args: {function_args}")
            content = available_functions[function_name](**function_args)
            tool_responses.append({
                'role': 'tool',
                'content': content,
            })
    return tool_responses

def call_gemma4(essay_id, essay_content, rag_engine):
    messages = [{
        'role': 'system',
        'content': """You are a judge at an essay writing competition. 
                        Asses the mastery of the writer by judging the quality of the essay, the reasoning, evidence, focus, coherence, vocabulary, 
                        and accuracy in grammar and sentence structure.
                        After assessing the mastery, use the available reference tool to determine the score of the essay.
                        Use the result of mastery assessment as query for the reference tool.
                        Answer in this format: essay_id, score. With score being in the numeric value only. Don't add prefix or suffix"""
    }, {
        'role': 'user',
        'content': f"Essay ID: {essay_id}\nEssay content: {essay_content}"
    }]

    # First call to let the model decide to use the tool
    response = ollama.chat(
        model='gemma4',
        messages=messages,
        tools=[REFERENCE_TOOL],
        options={"temperature": 0.3}, think=True
    )

    # Check for tool calls
    if response['message'].get('tool_calls'):
        messages.append(response['message'])
        tool_responses = handle_tool_calls(response['message']['tool_calls'], rag_engine)
        messages.extend(tool_responses)

        # Second call with tool results
        final_response = ollama.chat(
            model='gemma4',
            messages=messages,
            options={"temperature": 0.3}
        )
        print(final_response['message']['content'])
    else:
        print("Model did not call the tool.")
        print(response['message']['content'])

if __name__ == '__main__':
    rag_engine = RAGEngine(Path.cwd() / 'rag_docs')
    data = read_csv('./dataset/split/test.csv')
    essay_id, essay_content, score = data[0], data[1], data[2]
    call_gemma4(essay_id[20], essay_content[20], rag_engine)
    print(f"real score: {score[20]}")