# Ollama RAG Tool Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor static RAG retrieval into a dynamic tool-calling system using Ollama's native tool support in `src/main.py`.

**Architecture:** Initialize a Chroma-based RAG engine once at startup. Use `ollama.chat` with a tool definition for reference retrieval, enabling a "thought-action-observation" loop where the model retrieves scoring criteria as needed.

**Tech Stack:** `ollama` (Python SDK), `langchain-community`, `chroma`, `pandas`.

---

### Task 1: Refactor RAG Engine for Persistent Session Use

**Files:**
- Modify: `src/main.py`

- [ ] **Step 1: Move initialization logic to a reusable RAGEngine class**

```python
class RAGEngine:
    def __init__(self, doc_dir: Path):
        self.doc_dir = doc_dir
        self.db = self._initialize_db()

    def _initialize_db(self):
        loader = DirectoryLoader(self.doc_dir, glob="./*.txt", loader_cls=TextLoader)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=100)
        texts = text_splitter.split_documents(docs)
        embeddings = HuggingFaceEmbeddings(model="nomic-ai/nomic-embed-text-v1.5")
        return Chroma.from_documents(documents=texts, embedding=embeddings, collection_name="rag")

    def retrieve(self, query: str, k: int = 5):
        query_result = self.db.similarity_search(query, k=k)
        return "\n".join(doc.page_content for doc in query_result)
```

- [ ] **Step 2: Update `src/main.py` to use `RAGEngine` and remove redundant functions**
Remove: `document_loader`, `rag_embedding`, `rag_retrieval`.

- [ ] **Step 3: Verify initialization works**
Add a temporary test block to the end of `main.py` to confirm `RAGEngine` initializes and retrieves data.

---

### Task 2: Define Tool Schema and Handler

**Files:**
- Modify: `src/main.py`

- [ ] **Step 1: Define the tool schema for Ollama**

```python
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
```

- [ ] **Step 2: Implement the tool handler mapping**

```python
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
```

---

### Task 3: Implement Tool-Calling Loop in `call_gemma4`

**Files:**
- Modify: `src/main.py`

- [ ] **Step 1: Refactor `call_gemma4` to handle the iterative chat loop**

```python
def call_gemma4(essay_id, essay_content, rag_engine):
    messages = [{
        'role': 'system',
        'content': 'You are a judge at an essay writing competition. Use the available tool to retrieve scoring criteria before assigning a score. Answer in this format: essay_id, score'
    }, {
        'role': 'user',
        'content': f"Essay ID: {essay_id}\nEssay content: {essay_content}"
    }]

    # First call to let the model decide to use the tool
    response = ollama.chat(
        model='gemma4',
        messages=messages,
        tools=[REFERENCE_TOOL],
        options={"temperature": 0.3}
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
```

- [ ] **Step 2: Update `main` block to instantiate `RAGEngine` once**

```python
if __name__ == '__main__':
    rag_engine = RAGEngine(Path.cwd() / 'rag_docs')
    data = read_csv('./dataset/split/test.csv')
    essay_id, essay_content, score = data[0], data[1], data[2]
    call_gemma4(essay_id[3], essay_content[3], rag_engine)
    print(f"real score: {score[3]}")
```

---

### Task 4: Validation and Verification

- [ ] **Step 1: Run the script and observe output**
Run: `python src/main.py`
Expected: 
1. Log showing "Executing tool: retrieve_reference_info".
2. Model output in the requested format `essay_id, score`.
3. Compare with "real score".

- [ ] **Step 2: Refine the system prompt if the model skips tool calling**
If the model skips the tool, update the system prompt to be more forceful (e.g., "MANDATORY: You must call retrieve_reference_info before scoring").
