from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub  # or other LLM
from langchain_core.documents import Document
from langchain_community.llms import LlamaCpp  # For local models
import requests  # For Ollama API
from langchain_community.llms import HuggingFacePipeline  # Use community version
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.prompts import PromptTemplate  # Add PromptTemplate import
import os
import torch

def load_vectorstore():
    # Assuming you have previously saved to the local_db folder
    return FAISS.load_local(
        folder_path="vector_db",
        embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
        allow_dangerous_deserialization=True  # ✅ Explicitly allow pickle loading
    )

class OllamaAPI:
    """Simple wrapper for Ollama API"""
    def __init__(self, model="llama2", temperature=0.2):
        self.model = model
        self.temperature = temperature
        self.api_url = "http://localhost:11434/api/generate"
    
    def __call__(self, prompt):
        response = requests.post(
            self.api_url,
            json={
                "model": self.model,
                "prompt": prompt,
                "temperature": self.temperature
            }
        )
        return response.json()["response"]

def get_local_llm(model_type="huggingface", model_path=None):
    """
    Get locally deployed LLM model
    
    Parameters:
        model_type: Model type, options include "llama", "ollama", "huggingface"
        model_path: Model path, required for local models
    """
    if model_type == "llama":
        # Use LlamaCpp to load local model
        return LlamaCpp(
            model_path=model_path,  # e.g., "models/deepseek-llm-7b-chat.gguf"
            temperature=0.7,  # Increased temperature for more creative responses
            max_tokens=2048,  # Increased max tokens for longer responses
            n_ctx=2048,
            verbose=True
        )
    elif model_type == "ollama":
        # Use Ollama API
        return OllamaAPI(model="llama2", temperature=0.7)
    elif model_type == "huggingface":
        # Use HuggingFace local model
        model_id = model_path or "facebook/opt-125m"  # Use small causal model
        
        # Create model cache directory
        cache_dir = os.path.join(os.getcwd(), "model_cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        # Check if GPU is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
        
        # Load model, without 8-bit quantization to avoid bitsandbytes issues
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            trust_remote_code=True,
            offload_folder=cache_dir,  # Specify weight offload folder
            torch_dtype=torch.float16  # Use half precision
        )
        
        # Create pipeline with improved generation parameters
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=2048,  # Increased max length
            temperature=0.7,  # Increased temperature
            do_sample=True,
            top_p=0.95,
            top_k=50,
            repetition_penalty=1.2,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            truncation=True,  # Enable truncation
            return_full_text=False  # Don't return the prompt in the output
        )
        return HuggingFacePipeline(pipeline=pipe)
    else:
        # Default to HuggingFaceHub
        return HuggingFaceHub(
            repo_id="google/flan-t5-base",
            model_kwargs={"temperature": 0.7, "max_length": 2048}
        )

def ask_question(query: str, model_type="huggingface", model_path=None):
    db = load_vectorstore()

    # Get local LLM
    llm = get_local_llm(model_type=model_type, model_path=model_path)

    # Use custom prompt template to guide model to generate more natural responses
    prompt_template = PromptTemplate(
        template="""You are a professional medical assistant. Based on the following medical information, provide a clear and direct answer to the user's question. 
        Focus on practical steps and specific recommendations. If the information is insufficient, clearly state what additional information would be helpful.

        Medical Information:
        {context}

        User Question: {question}

        Provide a direct and helpful response:""",
        input_variables=["context", "question"]
    )
    
    # Create QA chain with custom prompt
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, 
        retriever=db.as_retriever(),
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": prompt_template
        }
    )
    
    # Get answer and source documents
    result = qa_chain.invoke(query)
    answer = result["result"]
    
    # Clean up the response
    answer = answer.replace("You are a professional medical assistant.", "").strip()
    answer = answer.replace("Medical Information:", "").strip()
    answer = answer.replace("User Question:", "").strip()
    answer = answer.replace("Provide a direct and helpful response:", "").strip()
    
    # Remove any retrieved information that might be in the response
    answer_lines = answer.split('\n')
    cleaned_lines = []
    for line in answer_lines:
        # Skip lines containing common unwanted content
        if any(marker in line for marker in [
            'Q:', 'A:', 'NIH:', 'Context:', 'Medical Information:', 
            'Based on the following', 'Note:', 'Question Title:',
            'Answer Text:', 'Name:', 'Email Address:', 'Comment:',
            'Please make sure', 'If you choose', 'We welcome',
            'Your name is required', 'will NOT be published'
        ]):
            continue
        # Skip empty lines and lines that are just formatting
        if line.strip() and not line.strip().startswith('*') and not line.strip().startswith('('):
            cleaned_lines.append(line)
    
    answer = '\n'.join(cleaned_lines).strip()
    
    # Ensure the response is not too short and contains actual advice
    if len(answer.strip()) < 100:  # If response is too short
        answer = "I apologize, but I need more information to provide a helpful response. Could you please provide more details about your question?"
    
    return answer

if __name__ == "__main__":
    # Set model type and path
    MODEL_TYPE = "huggingface"  # Changed to huggingface as Ollama requires model pulling first
    MODEL_PATH = "facebook/opt-1.3b"  # Use larger causal model
    
    print(f"Using model type: {MODEL_TYPE}")
    if MODEL_PATH:
        print(f"Model path: {MODEL_PATH}")
    
    while True:
        question = input("❓ Please enter your question (type 'exit' to quit):\n> ")
        if question.lower() in {"exit", "quit"}:
            break
        answer = ask_question(question, model_type=MODEL_TYPE, model_path=MODEL_PATH)
        print(f"\n💬 {answer}\n")
