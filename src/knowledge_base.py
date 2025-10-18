"""
Intelligent Knowledge Base with Vector Search
"""

import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Optional
import json
import pickle
import os

class KnowledgeBase:
    def __init__(self, config):
        self.config = config
        self.embedding_model = SentenceTransformer(config['knowledge']['embedding_model'])
        self.setup_vector_store()
    
    def setup_vector_store(self):
        """Setup ChromaDB vector store"""
        self.client = chromadb.PersistentClient(path="data/knowledge_base/vector_store")
        
        try:
            self.collection = self.client.get_collection("knowledge_base")
        except:
            self.collection = self.client.create_collection(
                name="knowledge_base",
                metadata={"hnsw:space": "cosine"}
            )
    
    def add_document(self, document: str, metadata: Dict = None):
        """Add document to knowledge base"""
        # Generate embedding
        embedding = self.embedding_model.encode([document])[0]
        
        # Add to collection
        doc_id = f"doc_{len(self.collection.get()['ids']) + 1}"
        self.collection.add(
            documents=[document],
            embeddings=[embedding.tolist()],
            metadatas=[metadata or {}],
            ids=[doc_id]
        )
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search knowledge base"""
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])[0]
            
            # Search in vector store
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )
            
            formatted_results = []
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i],
                    'score': 1 - results['distances'][0][i]  # Convert to similarity score
                })
            
            return formatted_results
        
        except Exception as e:
            print(f"Knowledge base search error: {e}")
            return []
    
    def add_web_results(self, query: str, search_results: List[Dict]):
        """Add web search results to knowledge base"""
        for result in search_results:
            content = f"Title: {result['title']}\nContent: {result['snippet']}\nSource: {result['url']}"
            metadata = {
                'source': 'web_search',
                'query': query,
                'timestamp': result['timestamp'],
                'url': result['url']
            }
            self.add_document(content, metadata)
    
    def get_context(self, query: str, max_tokens: int = 2000) -> str:
        """Get relevant context for query"""
        results = self.search(query)
        
        context_parts = []
        current_length = 0
        
        for result in results:
            if result['score'] >= self.config['knowledge']['similarity_threshold']:
                content = f"Source: {result['metadata'].get('source', 'unknown')}\n{result['content']}\n\n"
                content_length = len(content)
                
                if current_length + content_length <= max_tokens:
                    context_parts.append(content)
                    current_length += content_length
                else:
                    break
        
        return "\n".join(context_parts)