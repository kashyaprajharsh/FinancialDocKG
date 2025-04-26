
from sentence_transformers import SentenceTransformer
import torch
from typing import List
import numpy as np



# Initialize sentence transformer model with error handling
def init_sentence_transformer():
    try:
        # Check if CUDA is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        model = SentenceTransformer('BAAI/bge-small-en-v1.5')
        model.to(device)
        return model
    except Exception as e:
        print(f"Error initializing SentenceTransformer: {e}")
        print("Falling back to simpler model...")
        try:
            model = SentenceTransformer('all-MiniLM-L6-v2')
            model.to(device)
            return model
        except Exception as e:
            print(f"Critical error initializing SentenceTransformer: {e}")
            raise



try:
    sentence_model = init_sentence_transformer()
    print("Successfully initialized SentenceTransformer model")
except Exception as e:
    print(f"Failed to initialize SentenceTransformer: {e}")
    raise


def get_embedding(text: str, task_type: str = "SEMANTIC_SIMILARITY") -> List[float]:
    """
    Get embedding for a text using SentenceTransformer.
    
    Args:
        text: Text to embed
        task_type: Type of embedding task (ignored for SentenceTransformer)
        
    Returns:
        List of floats representing the embedding
    """
    try:
        text = str(text).strip()
        if not text:
            print("Warning: Empty text provided for embedding")
            return None
            
        device = next(sentence_model.parameters()).device
        
        with torch.no_grad():  
            embedding = sentence_model.encode(
                text,
                convert_to_tensor=True,
                device=device
            )
            return embedding.cpu().numpy().tolist()
    except Exception as e:
        print(f"Error getting embedding: {str(e)}")
        return None

def get_embeddings_batch(texts: List[str], task_type: str) -> List[List[float]]:
    """
    Get embeddings for a batch of texts using SentenceTransformer.
    
    Args:
        texts: List of texts to embed
        task_type: Type of embedding task (ignored for SentenceTransformer)
        
    Returns:
        List of embedding vectors
    """
    try:
        texts = [str(text).strip() for text in texts]
        valid_texts = [text for text in texts if text]
        
        if not valid_texts:
            print("Warning: No valid texts provided for embedding")
            return []
            
        device = next(sentence_model.parameters()).device
        
        with torch.no_grad():  
            embeddings = sentence_model.encode(
                valid_texts,
                convert_to_tensor=True,
                device=device,
                batch_size=32  
            )
            return embeddings.cpu().numpy().tolist()
    except Exception as e:
        print(f"Error getting batch embeddings: {str(e)}")
        embedding_dim = sentence_model.get_sentence_embedding_dimension()
        return [[0] * embedding_dim for _ in texts]

def compute_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    """
    Compute cosine similarity between two embeddings.
    """
    a = np.array(embedding1)
    b = np.array(embedding2)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))