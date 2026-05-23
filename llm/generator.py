import ollama
from llm.guardrails import is_safe_query, is_grounded_answer

# The model we're using locally via Ollama
MODEL_NAME = "mistral"


def build_prompt(question, retrieved_chunks):
    """
    Builds the full prompt that gets sent to the LLM.

    The prompt does three things:
    1. Tells the LLM exactly what role it plays
    2. Gives it the retrieved context chunks to read from
    3. Asks the question and tells it to ONLY use the context

    This is the core of RAG — the LLM reads your documents,
    it doesn't rely on its own memory.
    """
    # Build the context block from retrieved chunks
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks):
        context_parts.append(
            f"[Source {i+1}] File: {chunk['file_name']}\n"
            f"{chunk['text']}"
        )

    context_block = "\n\n---\n\n".join(context_parts)

    # The full prompt
    prompt = f"""You are a helpful assistant that answers questions strictly based on the provided context documents.

RULES:
1. Answer ONLY using information found in the context below.
2. If the answer is not in the context, say exactly: "I don't know based on the provided documents."
3. Always be concise and factual.
4. Do not make up information or use your own knowledge.
5. After your answer, list which sources you used.

CONTEXT:
{context_block}

QUESTION: {question}

ANSWER:"""

    return prompt


def generate_answer(question, retrieved_chunks):
    """
    Takes a question and retrieved chunks,
    builds the prompt, calls Ollama, returns
    a structured response with answer + citations.
    """
    prompt = build_prompt(question, retrieved_chunks)

    # Call the local Ollama model
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    answer_text = response['message']['content'].strip()

    # Build citation objects from the retrieved chunks
    citations = []
    for chunk in retrieved_chunks:
        citations.append({
            "file_name": chunk['file_name'],
            "drive_url": chunk['drive_url'],
            "snippet":   chunk['text'][:200] + "..."
        })

    return {
        "answer":    answer_text,
        "citations": citations
    }


def query_rag(question, mode="hybrid", top_k=5):
    """
    The main end-to-end RAG function.
    This is what the API will call.

    Flow:
    1. Check guardrails — is the question safe?
    2. Retrieve relevant chunks from Elasticsearch
    3. Generate a grounded answer using the LLM
    4. Return answer + citations
    """
    # Import here to avoid circular imports
    from retrieval.retriever import retrieve

    print(f"\n{'='*50}")
    print(f"Question: {question}")
    print(f"Mode: {mode}, Top-k: {top_k}")

    # Step 1: Guardrail check
    safe, reason = is_safe_query(question)
    if not safe:
        print(f"BLOCKED by guardrail: {reason}")
        return {
            "answer":    f"I cannot answer this question. {reason}",
            "citations": [],
            "blocked":   True,
            "reason":    reason
        }

    # Step 2: Retrieve relevant chunks
    print("Retrieving chunks...")
    chunks = retrieve(question, mode=mode, top_k=top_k)

    if not chunks:
        return {
            "answer":    "I don't know based on the provided documents. No relevant content was found.",
            "citations": [],
            "blocked":   False
        }

    # Step 3: Generate answer
    print("Generating answer with LLM...")
    result = generate_answer(question, chunks)
    result["blocked"] = False

    print(f"Answer generated ({len(result['answer'])} chars)")
    return result