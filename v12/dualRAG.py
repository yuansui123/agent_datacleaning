"""
Dual-Modal Retrieval System for Example Bank

Usage:
    retriever = DualRAGRetriever("path/to/example_bank")
    
    # Text-based retrieval
    results = retriever.search_by_text(
        "Muscle artifact with signal thickening at 0.5-1.0s",
        top_k=5
    )
    
    # Image-based retrieval
    results = retriever.search_by_image(
        "path/to/query_image.png",
        top_k=5
    )
    
    # Hybrid retrieval (text + image)
    results = retriever.search_hybrid(
        text="Muscle artifact",
        image_path="path/to/query.png",
        top_k_text=3,
        top_k_vision=2,
        text_weight=0.6,
        vision_weight=0.4
    )
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Literal
from dataclasses import dataclass
import matplotlib.pyplot as plt
from PIL import Image


@dataclass
class RetrievalResult:
    """Single retrieval result with metadata."""
    example_id: str
    score: float
    plot_type: str
    category: str
    image_path: str
    description: str
    source_metadata: Dict[str, Any]
    modality: Literal["text", "vision", "hybrid"]  # How it was retrieved


class DualRAGRetriever:
    """Dual-modal retrieval system for example banks."""
    
    def __init__(
        self,
        example_bank_dir: str,
        example_bank_file: str = "example_bank.jsonl",
        config_file: str = "config.json",
        device: str = "auto"
    ):
        """
        Initialize retriever.
        
        Args:
            example_bank_dir: Directory containing example bank
            example_bank_file: Name of JSONL file
            config_file: Name of config file
            device: "cuda", "cpu", or "auto"
        """
        self.example_bank_dir = Path(example_bank_dir)
        
        # Load config
        config_path = self.example_bank_dir / config_file
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        
        with open(config_path, "r") as f:
            self.config = json.load(f)
        
        print(f"Loading example bank...")
        print(f"  Embedder type: {self.config['embedder_type']}")
        print(f"  Embedding dim: {self.config['embedding_dim']}")
        print(f"  Num examples: {self.config['num_examples']}")
        
        # Load examples
        self.examples = self._load_examples(self.example_bank_dir / example_bank_file)
        
        # Build embedding matrices
        self.text_embeddings, self.vision_embeddings = self._build_embedding_matrices()
        
        # Initialize embedder for query encoding
        self.embedder = self._init_embedder(self.config['embedder_type'], device)
        
        print(f"✓ Loaded {len(self.examples)} examples\n")
    
    def _load_examples(self, jsonl_path: Path) -> List[Dict[str, Any]]:
        """Load examples from JSONL."""
        examples = []
        with open(jsonl_path, "r") as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line))
        return examples
    
    def _build_embedding_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """Build embedding matrices for fast similarity search."""
        text_embs = []
        vision_embs = []
        
        for ex in self.examples:
            if ex.get('text_embedding'):
                text_embs.append(ex['text_embedding'])
            else:
                text_embs.append([0.0] * self.config['embedding_dim'])
            
            if ex.get('vision_embedding'):
                vision_embs.append(ex['vision_embedding'])
            else:
                vision_embs.append([0.0] * self.config['embedding_dim'])
        
        text_matrix = np.array(text_embs, dtype=np.float32)
        vision_matrix = np.array(vision_embs, dtype=np.float32)
        
        # Normalize for cosine similarity
        text_matrix = text_matrix / (np.linalg.norm(text_matrix, axis=1, keepdims=True) + 1e-8)
        vision_matrix = vision_matrix / (np.linalg.norm(vision_matrix, axis=1, keepdims=True) + 1e-8)
        
        return text_matrix, vision_matrix
    
    def _init_embedder(self, embedder_type: str, device: str):
        """Initialize embedder for query encoding."""
        if embedder_type == "clip":
            from transformers import CLIPProcessor, CLIPModel
            import torch
            
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            
            model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
            model.to(device)
            model.eval()
            
            return {"type": "clip", "model": model, "processor": processor, "device": device}
        
        elif embedder_type == "gemini":
            raise NotImplementedError("Gemini retrieval not yet implemented")
        
        elif embedder_type == "openai":
            raise NotImplementedError("OpenAI retrieval not yet implemented")
        
        else:
            raise ValueError(f"Unknown embedder type: {embedder_type}")
    
    def encode_text(self, text: str) -> np.ndarray:
        """Encode text query to embedding."""
        if self.embedder["type"] == "clip":
            import torch
            
            # Truncate text to CLIP's limit
            tokens = self.embedder["processor"].tokenizer.encode(text)
            if len(tokens) > 77:
                tokens = tokens[:75]  # Leave room for special tokens
                text = self.embedder["processor"].tokenizer.decode(tokens, skip_special_tokens=True)
            
            inputs = self.embedder["processor"](
                text=text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77
            ).to(self.embedder["device"])
            
            with torch.no_grad():
                features = self.embedder["model"].get_text_features(**inputs)
                features = features / features.norm(dim=-1, keepdim=True)
            
            return features.cpu().numpy()[0]
        
        else:
            raise NotImplementedError(f"Text encoding not implemented for {self.embedder['type']}")
    
    def encode_image(self, image_path: str) -> np.ndarray:
        """Encode image query to embedding."""
        if self.embedder["type"] == "clip":
            import torch
            
            image = Image.open(image_path).convert("RGB")
            inputs = self.embedder["processor"](
                images=image,
                return_tensors="pt"
            ).to(self.embedder["device"])
            
            with torch.no_grad():
                features = self.embedder["model"].get_image_features(**inputs)
                features = features / features.norm(dim=-1, keepdim=True)
            
            return features.cpu().numpy()[0]
        
        else:
            raise NotImplementedError(f"Image encoding not implemented for {self.embedder['type']}")
    
    def search_by_text(
        self,
        query: str,
        top_k: int = 5,
        filter_category: Optional[str] = None,
        filter_plot_type: Optional[str] = None
    ) -> List[RetrievalResult]:
        """
        Search by text description.
        
        Args:
            query: Text description
            top_k: Number of results to return
            filter_category: Only return examples from this category
            filter_plot_type: Only return examples of this plot type
        
        Returns:
            List of RetrievalResult
        """
        # Encode query
        query_embedding = self.encode_text(query)
        
        # Compute similarities
        similarities = np.dot(self.text_embeddings, query_embedding)
        
        # Apply filters
        valid_indices = self._apply_filters(filter_category, filter_plot_type)
        similarities[~valid_indices] = -np.inf
        
        # Get top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Build results
        results = []
        for idx in top_indices:
            if similarities[idx] > -np.inf:
                results.append(self._build_result(idx, similarities[idx], "text"))
        
        return results
    
    def search_by_image(
        self,
        image_path: str,
        top_k: int = 5,
        filter_category: Optional[str] = None,
        filter_plot_type: Optional[str] = None
    ) -> List[RetrievalResult]:
        """
        Search by image similarity.
        
        Args:
            image_path: Path to query image
            top_k: Number of results to return
            filter_category: Only return examples from this category
            filter_plot_type: Only return examples of this plot type
        
        Returns:
            List of RetrievalResult
        """
        # Encode query
        query_embedding = self.encode_image(image_path)
        
        # Compute similarities
        similarities = np.dot(self.vision_embeddings, query_embedding)
        
        # Apply filters
        valid_indices = self._apply_filters(filter_category, filter_plot_type)
        similarities[~valid_indices] = -np.inf
        
        # Get top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Build results
        results = []
        for idx in top_indices:
            if similarities[idx] > -np.inf:
                results.append(self._build_result(idx, similarities[idx], "vision"))
        
        return results
    
    def search_hybrid(
        self,
        text: Optional[str] = None,
        image_path: Optional[str] = None,
        top_k: int = 5,
        text_weight: float = 0.5,
        vision_weight: float = 0.5,
        filter_category: Optional[str] = None,
        filter_plot_type: Optional[str] = None,
        fusion_method: Literal["weighted_sum", "max", "min"] = "weighted_sum"
    ) -> List[RetrievalResult]:
        """
        Hybrid search using both text and image.
        
        Args:
            text: Text query (optional)
            image_path: Image query (optional)
            top_k: Number of results
            text_weight: Weight for text similarity
            vision_weight: Weight for vision similarity
            filter_category: Category filter
            filter_plot_type: Plot type filter
            fusion_method: How to combine scores
        
        Returns:
            List of RetrievalResult
        """
        if text is None and image_path is None:
            raise ValueError("Must provide at least one of text or image_path")
        
        # Compute similarities
        text_sim = None
        vision_sim = None
        
        if text:
            query_text_emb = self.encode_text(text)
            text_sim = np.dot(self.text_embeddings, query_text_emb)
        
        if image_path:
            query_vision_emb = self.encode_image(image_path)
            vision_sim = np.dot(self.vision_embeddings, query_vision_emb)
        
        # Fuse scores
        if fusion_method == "weighted_sum":
            if text_sim is not None and vision_sim is not None:
                combined_sim = text_weight * text_sim + vision_weight * vision_sim
            elif text_sim is not None:
                combined_sim = text_sim
            else:
                combined_sim = vision_sim
        
        elif fusion_method == "max":
            if text_sim is not None and vision_sim is not None:
                combined_sim = np.maximum(text_sim, vision_sim)
            elif text_sim is not None:
                combined_sim = text_sim
            else:
                combined_sim = vision_sim
        
        elif fusion_method == "min":
            if text_sim is not None and vision_sim is not None:
                combined_sim = np.minimum(text_sim, vision_sim)
            elif text_sim is not None:
                combined_sim = text_sim
            else:
                combined_sim = vision_sim
        
        # Apply filters
        valid_indices = self._apply_filters(filter_category, filter_plot_type)
        combined_sim[~valid_indices] = -np.inf
        
        # Get top-k
        top_indices = np.argsort(combined_sim)[::-1][:top_k]
        
        # Build results
        results = []
        for idx in top_indices:
            if combined_sim[idx] > -np.inf:
                results.append(self._build_result(idx, combined_sim[idx], "hybrid"))
        
        return results
    
    def _apply_filters(
        self,
        filter_category: Optional[str],
        filter_plot_type: Optional[str]
    ) -> np.ndarray:
        """Apply category and plot type filters."""
        valid = np.ones(len(self.examples), dtype=bool)
        
        if filter_category:
            for i, ex in enumerate(self.examples):
                if ex.get('source_metadata', {}).get('category') != filter_category:
                    valid[i] = False
        
        if filter_plot_type:
            for i, ex in enumerate(self.examples):
                if ex.get('plot_type') != filter_plot_type:
                    valid[i] = False
        
        return valid
    
    def _build_result(self, idx: int, score: float, modality: str) -> RetrievalResult:
        """Build RetrievalResult from example index."""
        ex = self.examples[idx]
        
        return RetrievalResult(
            example_id=ex['id'],
            score=float(score),
            plot_type=ex['plot_type'],
            category=ex.get('source_metadata', {}).get('category', 'unknown'),
            image_path=ex['image_path'],
            description=ex.get('llm_description', ''),
            source_metadata=ex.get('source_metadata', {}),
            modality=modality
        )
    
    def visualize_results(
        self,
        results: List[RetrievalResult],
        query_text: Optional[str] = None,
        query_image_path: Optional[str] = None,
        figsize: Tuple[int, int] = (15, 3)
    ):
        """Visualize retrieval results."""
        n_results = len(results)
        
        fig, axes = plt.subplots(1, n_results + 1, figsize=figsize)
        
        # Plot query
        ax = axes[0]
        if query_image_path and os.path.exists(query_image_path):
            img = plt.imread(query_image_path)
            ax.imshow(img)
            ax.set_title(f"Query\n{query_text[:30] if query_text else ''}", fontsize=10)
        else:
            ax.text(0.5, 0.5, f"Query:\n{query_text}", ha='center', va='center', wrap=True)
            ax.set_title("Text Query", fontsize=10)
        ax.axis('off')
        
        # Plot results
        for i, result in enumerate(results):
            ax = axes[i + 1]
            if os.path.exists(result.image_path):
                img = plt.imread(result.image_path)
                ax.imshow(img)
            ax.set_title(
                f"#{i+1} (score: {result.score:.3f})\n{result.category} - {result.plot_type}",
                fontsize=9
            )
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()


# ============================================================
# Example Usage
# ============================================================

if __name__ == "__main__":
    # Initialize retriever
    retriever = DualRAGRetriever("path/to/example_bank")
    
    # Text search
    print("\n=== Text Search ===")
    results = retriever.search_by_text(
        "Muscle artifact with signal thickening",
        top_k=5,
        filter_category="noise"
    )
    
    for i, result in enumerate(results, 1):
        print(f"{i}. {result.example_id} (score: {result.score:.3f})")
        print(f"   Category: {result.category}, Plot: {result.plot_type}")
        print(f"   Description: {result.description[:100]}...")
        print()
    
    # Image search
    print("\n=== Image Search ===")
    results = retriever.search_by_image(
        "path/to/query_image.png",
        top_k=5
    )
    
    retriever.visualize_results(
        results,
        query_image_path="path/to/query_image.png"
    )
    
    # Hybrid search
    print("\n=== Hybrid Search ===")
    results = retriever.search_hybrid(
        text="Pathological spikes",
        image_path="path/to/query_image.png",
        top_k=5,
        text_weight=0.6,
        vision_weight=0.4
    )