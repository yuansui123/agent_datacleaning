"""
Example Bank Generation Module
generating example banks from time series data with dual text+vision embeddings.

Usage in notebook:
    from example_bank import ExampleBankPipeline
    
    pipeline = ExampleBankPipeline(
        tools_registry=tools.TOOL_REGISTRY,
        output_dir="output/my_bank",
    )
    
    example_bank_path = pipeline.generate_from_dataset(
        dataset=dataset,
        categories=['physiology'],
        k_per_category=5,
    )
"""

import os
import json
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Callable, Literal
from dataclasses import dataclass, asdict
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


# ============================================================
# Helper Functions
# ============================================================

def _convert_numpy_types(obj):
    """
    Recursively convert numpy types to Python native types for JSON serialization.
    
    Args:
        obj: Any object that might contain numpy types
    
    Returns:
        Object with numpy types converted to Python native types
    """
    if isinstance(obj, dict):
        return {key: _convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(_convert_numpy_types(item) for item in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


# ============================================================
# Data Structures
# ============================================================

@dataclass
class TimeSeriesData:
    """Input time series with metadata."""
    source_id: str
    data: np.ndarray  # (n_channels, n_timepoints)
    sampling_rate: float
    metadata: Dict[str, Any]


@dataclass
class PlotConfig:
    """Configuration for a plot type."""
    tool_name: str
    params: Dict[str, Any]
    plot_type_label: str


@dataclass
class ExampleMetadata:
    """Complete metadata for a single example."""
    id: str
    source_id: str
    plot_type: str
    tool: Dict[str, Any]
    source_metadata: Dict[str, Any]
    
    llm_description: Optional[str] = None
    
    image_path: str = ""
    text_embedding: Optional[List[float]] = None
    vision_embedding: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        # Convert numpy types to Python native types
        return _convert_numpy_types(data)


# ============================================================
# Plot Generator
# ============================================================

class PlotGenerator:
    """
    Generates plots using tools that return either:
    - (fig, metadata) tuples (new API)
    - dict with file_path (old API)
    """
    
    def __init__(self, tools_registry: Dict[str, Callable], output_dir: str):
        self.tools_registry = tools_registry
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_figure(self, fig: plt.Figure, prefix: str, dpi: int = 150) -> str:
        """Save a matplotlib figure and return path."""
        plot_id = f"{prefix}_{uuid.uuid4().hex[:8]}"
        filename = f"{plot_id}.png"
        file_path = self.output_dir / filename
        
        fig.savefig(file_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        
        return str(file_path)
    
    def generate_plot(
        self,
        ts_data: TimeSeriesData,
        plot_config: PlotConfig,
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate a single plot."""
        tool_fn = self.tools_registry.get(plot_config.tool_name)
        if tool_fn is None:
            raise ValueError(f"Tool {plot_config.tool_name} not found")
        
        # Call tool
        result = tool_fn(
            data=ts_data.data,
            sampling_rate=ts_data.sampling_rate,
            **plot_config.params
        )
        
        # Handle both new and old API
        if isinstance(result, tuple) and len(result) == 2:
            # New API: (fig, metadata)
            fig, metadata = result
            image_path = self.save_figure(fig, prefix=plot_config.plot_type_label)
        else:
            # Old API: dict with file_path
            metadata = result
            image_path = result.get("file_path", "")
        
        return image_path, metadata
    
    def generate_all_plots(
        self,
        ts_data: TimeSeriesData,
        plot_configs: List[PlotConfig],
        verbose: bool = True,
    ) -> List[ExampleMetadata]:
        """Generate all configured plots for a time series."""
        examples = []
        
        for plot_config in plot_configs:
            try:
                if verbose:
                    print(f"    Generating {plot_config.plot_type_label}...")
                
                image_path, plot_metadata = self.generate_plot(ts_data, plot_config)
                
                example = ExampleMetadata(
                    id=f"{ts_data.source_id}_{plot_config.plot_type_label}",
                    source_id=ts_data.source_id,
                    plot_type=plot_config.plot_type_label,
                    tool={
                        "name": plot_config.tool_name,
                        "params": plot_config.params,
                    },
                    source_metadata=ts_data.metadata,
                    image_path=image_path,
                )
                
                examples.append(example)
                
                if verbose:
                    print(f"      ✓ {image_path}")
                
            except Exception as e:
                if verbose:
                    print(f"      ✗ Failed: {e}")
                continue
        
        return examples


# ============================================================
# LLM Describer
# ============================================================

class LLMDescriber:
    """Generates semantic descriptions using LLM vision."""
    
    def __init__(self, api_key: str, model: str = "gpt-4o", max_tokens: int = 60):  
        if OpenAI is None:
            raise ImportError("OpenAI package not installed. Run: pip install openai")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens  # maximum tokens in response
    
    def describe_example(
        self,
        example: ExampleMetadata,
        custom_prompt: Optional[str] = None,
    ) -> Tuple[str, str]:  
        """Generate LLM description for an example."""
        import base64
        
        if not os.path.exists(example.image_path):
            raise FileNotFoundError(f"Image not found: {example.image_path}")
        
        # Encode image
        with open(example.image_path, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode('utf-8')
        
        # Build prompt
        if custom_prompt is None:
            custom_prompt = self._build_default_prompt(example)
        
        # Call LLM
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": custom_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                        }
                    ]
                }
            ],
            max_tokens=self.max_tokens,
            temperature=0.3,  # Lower temperature for more consistent, focused outputs
        )
        
        response_text = response.choices[0].message.content

        # #print this image for debug
        # print("Debug: LLM response:")
        # print(response_text)

        # #plot image for debug
        # import matplotlib.pyplot as plt
        # img = plt.imread(example.image_path)
        # plt.imshow(img)
        # plt.axis('off')
        # plt.show()

         # Parse response and combine description + reasoning
        try:
            import re
            # Remove markdown code fences
            clean_text = re.sub(r"```(?:json)?", "", response_text).strip("`").strip()
            result = json.loads(clean_text)
            
            # Combine description and reasoning into single string
            description = result.get("description", "")
            reasoning = result.get("reasoning", "")
            
            combined = f"{description} {reasoning}".strip()
            return combined
            
        except json.JSONDecodeError:
            # If JSON parsing fails, just return cleaned text
            return response_text.replace("```json", "").replace("```", "").strip()
        
    def _build_default_prompt(self, example: ExampleMetadata) -> str:
        """Build default prompt for LLM description."""
    
        return f"""You are analyzing a neural electrophysiology plot to create a reference example.

PLOT INFORMATION:
- Type: {example.plot_type}
- Tool: {example.tool['name']}
- Category: {example.source_metadata.get('category', 'N/A')}

CATEGORY DEFINITIONS:
- powerline_60hz: 60Hz electrical interference
  - Spectrogram: thin horizontal line at 60Hz
  - Time series: regular oscillatory pattern
  - PSD: sharp peak at 60Hz
  - Power density matrix: horizontal band at 60Hz

- pathology: Abnormal neural activity
  - **Best observed in power density matrix plots**
  - Power density matrix: vertical lines or bands indicating transient high-power events

- noise: Non-neural artifacts (excluding powerline)
  - **Best observed in time series plots**
  - Time series: look for irregular dense patches or amplitude anomalies
    - Muscle artifacts: signal thickening, high-amplitude bursts in specific time windows
    - Movement artifacts: baseline shifts, abrupt changes
    - Electrode artifacts: spikes, discontinuities

- physiology: Normal neural activity
  - Power density matrix: smooth patterns without abrupt vertical features

YOUR TASK:
Write a concise sentence (max 30 words):
What visual features distinguish this signal? Focus on patterns NOT already stated in metadata (plot type/tool). 
Why does this example belong to the "{example.source_metadata.get('category', 'N/A')}" category? Connect visual features to category criteria.

REQUIREMENTS:
- Be specific with time intervals and frequency bands
- Use technical terms accurately (amplitude, frequency, baseline)
- Avoid redundancy with metadata
- Stay factual, no speculation

Respond in valid JSON format:
{{"description": "..."}}"""
    
    def describe_batch(
        self,
        examples: List[ExampleMetadata],
        custom_prompt: Optional[str] = None,
        verbose: bool = True,
    ) -> List[ExampleMetadata]:
        """Generate descriptions for multiple examples."""
        iterator = tqdm(examples, desc="LLM descriptions") if verbose else examples
        
        for example in iterator:
            try:
                desc = self.describe_example(example, custom_prompt)
                example.llm_description = desc
                # Removed confidence assignment
            except Exception as e:
                if verbose:
                    print(f"Warning: Failed to describe {example.id}: {e}")
                example.llm_description = str(e)
        
        return examples


# ============================================================
# Embedding Computer
# ============================================================

class EmbeddingComputer:
    """Computes text and vision embeddings."""
    
    def __init__(self, embedder_type: str = "clip", api_key: Optional[str] = None):
        self.embedder_type = embedder_type
        
        if embedder_type == "clip":
            from transformers import CLIPProcessor, CLIPModel
            import torch
            
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Loading CLIP on {self.device}...")
            
            self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
            self.model.to(self.device)
            self.model.eval()
            
        elif embedder_type == "gemini":
            try:
                import google.generativeai as genai
            except ImportError:
                raise ImportError("Google AI package required. Run: pip install google-generativeai")
            if not api_key:
                raise ValueError("api_key required for embedder_type='gemini'")
            genai.configure(api_key=api_key)
            self.genai = genai
            print("Initialized Gemini embeddings...")
            
        elif embedder_type == "openai":
            if OpenAI is None:
                raise ImportError("OpenAI package required. Run: pip install openai")
            if not api_key:
                raise ValueError("api_key required for embedder_type='openai'")
            self.client = OpenAI(api_key=api_key)
            print("Initialized OpenAI embeddings (Note: Uses text description of images, not true vision embeddings)...")
        else:
            raise ValueError("embedder_type must be 'clip', 'gemini', or 'openai'")
    
    def get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        if self.embedder_type == "clip":
            import torch
            inputs = self.processor(text=text, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            return text_features.cpu().numpy()[0].tolist()
        
        elif self.embedder_type == "gemini":
            result = self.genai.embed_content(
                model='models/text-embedding-004',
                content=text,
                task_type="retrieval_document"
            )
            return result['embedding']
        
        else:  # openai
            response = self.client.embeddings.create(
                model="text-embedding-3-large",
                input=text
            )
            return response.data[0].embedding
    
    def get_vision_embedding(self, image_path: str) -> List[float]:
        """Get vision embedding."""
        from PIL import Image
        
        if self.embedder_type == "clip":
            import torch
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            return image_features.cpu().numpy()[0].tolist()
        
        elif self.embedder_type == "gemini":
            # Gemini supports true multimodal embeddings
            image = Image.open(image_path)
            
            result = self.genai.embed_content(
                model='models/text-embedding-004',  # Supports multimodal
                content=image,
                task_type="retrieval_document"
            )
            return result['embedding']
        
        else:  # openai
            # WARNING: OpenAI doesn't provide direct vision embeddings
            # This is a workaround: image → text description → text embedding
            # NOT true vision embeddings!
            import base64
            with open(image_path, "rb") as f:
                base64_image = base64.b64encode(f.read()).decode('utf-8')
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe the visual features, patterns, colors, and structure in this plot in detail. Focus on what you SEE, not interpretation."},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                    ]
                }],
                max_tokens=300,
            )
            description = response.choices[0].message.content
            # Convert description to embedding (semantic, not visual!)
            return self.get_text_embedding(description)
    
    def build_text_input(self, example: ExampleMetadata) -> str:
        """Build text input from example metadata."""
        parts = [
            f"Plot Type: {example.plot_type}",
            #f"Parameters: {json.dumps(example.tool['params'])}",
        ]
        
        if example.source_metadata:
            parts.append(f"Category: {json.dumps(example.source_metadata['category'])}")
        
        if example.llm_description:
            parts.append(f"Description: {example.llm_description}")
        
        return "\n".join(parts)
    
    def compute_embeddings(
        self,
        examples: List[ExampleMetadata],
        compute_text: bool = True,
        compute_vision: bool = True,
        verbose: bool = True,
    ) -> List[ExampleMetadata]:
        """Compute embeddings for all examples."""
        iterator = tqdm(examples, desc="Computing embeddings") if verbose else examples
        
        for example in iterator:
            try:
                if compute_text:
                    text_input = self.build_text_input(example)

                    # #print text input for debug
                    # print("Debug: Text input for embedding:")
                    # print(text_input)
                    
                    # Safeguard: Truncate if text exceeds token limit
                    text_input = self._truncate_text_if_needed(text_input, verbose=verbose, example_id=example.id)
                    
                    example.text_embedding = self.get_text_embedding(text_input)
                
                if compute_vision:
                    if os.path.exists(example.image_path):
                        example.vision_embedding = self.get_vision_embedding(example.image_path)
                    elif verbose:
                        print(f"Warning: Image not found for {example.id}")
            
            except TypeError as e:
                if "JSON serializable" in str(e):
                    if verbose:
                        print(f"Warning: Metadata contains non-serializable types for {example.id}")
                        print(f"  Converting numpy types to Python native types...")
                    # Try to convert and re-attempt
                    try:
                        example.source_metadata = _convert_numpy_types(example.source_metadata)
                        example.tool = _convert_numpy_types(example.tool)
                        # Retry
                        if compute_text:
                            text_input = self.build_text_input(example)
                            text_input = self._truncate_text_if_needed(text_input, verbose=verbose, example_id=example.id)
                            example.text_embedding = self.get_text_embedding(text_input)
                        if compute_vision and os.path.exists(example.image_path):
                            example.vision_embedding = self.get_vision_embedding(example.image_path)
                    except Exception as e2:
                        if verbose:
                            print(f"Warning: Still failed for {example.id}: {e2}")
                else:
                    if verbose:
                        print(f"Warning: Failed embeddings for {example.id}: {e}")
            except Exception as e:
                if verbose:
                    print(f"Warning: Failed embeddings for {example.id}: {e}")
                continue
        
        return examples

    def _truncate_text_if_needed(self, text: str, verbose: bool = False, example_id: str = None) -> str:
        """Truncate text if it exceeds the embedder's token limit."""
        if self.embedder_type == "clip":
            max_tokens = 77
        elif self.embedder_type == "gemini":
            max_tokens = 2048
        elif self.embedder_type == "openai":
            max_tokens = 8191
        else:
            # Unknown embedder, be conservative
            max_tokens = 77
        
        # Rough estimate: 1 token ≈ 4 characters (conservative)
        # For more accurate counting, you'd need the actual tokenizer
        estimated_tokens = len(text) // 4
        
        if estimated_tokens > max_tokens:
            if verbose:
                print(f"Warning: Text for {example_id} exceeds {max_tokens} tokens (~{estimated_tokens} estimated). Truncating...")
            
            # Truncate to approximately max_tokens
            # Keep a safety margin (use 90% of limit)
            safe_char_limit = int(max_tokens * 4 * 0.9)
            truncated_text = text[:safe_char_limit] + "..."
            
            if verbose:
                print(f"  Original length: {len(text)} chars, Truncated to: {len(truncated_text)} chars")
            
            return truncated_text
        
        return text


# ============================================================
# Main Pipeline
# ============================================================

class ExampleBankPipeline:
    """
    Main pipeline for example bank generation.
    
    Usage:
        pipeline = ExampleBankPipeline(
            tools_registry=tools.TOOL_REGISTRY,
            output_dir="output/my_bank",
            embedder_type="clip",
        )
        
        bank_path = pipeline.generate_from_dataset(
            dataset=dataset,
            categories=['noise'],
            k_per_category=5,
        )
    """
    
    def __init__(
        self,
        tools_registry: Dict[str, Callable],
        output_dir: str,
        embedder_type: Literal["clip", "openai"] = "clip",
        api_key: Optional[str] = None,
    ):
        """
        Initialize the pipeline.
        
        Args:
            tools_registry: Dictionary of plotting tools (tools.TOOL_REGISTRY)
            output_dir: Output directory for plots and example bank
            embedder_type: "clip" (local, free) or "openai" (API, costs money)
            api_key: OpenAI API key (for LLM descriptions or openai embedder)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.plots_dir = self.output_dir / "plots"
        
        # Initialize components
        self.plot_generator = PlotGenerator(tools_registry, str(self.plots_dir))
        self.embedding_computer = EmbeddingComputer(embedder_type, api_key)
        
        self._llm_describer = None
        self._api_key = api_key
    
    @property
    def llm_describer(self) -> LLMDescriber:
        """Lazy initialization of LLM describer."""
        if self._llm_describer is None:
            if not self._api_key:
                raise ValueError("api_key required for LLM descriptions")
            self._llm_describer = LLMDescriber(self._api_key)
        return self._llm_describer
    
    def load_from_dataset(
        self,
        dataset,
        categories: List[str],
        k_per_category: int,
        category_col: str = 'category_name',
        sampling_rate_key: str = 'sampling_rate',
        default_sampling_rate: float = 5000.0,
    ) -> List[TimeSeriesData]:
        """
        Load time series from a dataset object.
        
        Args:
            dataset: Dataset with get_random_k_segments_by_category() method
            categories: List of category values to sample
            k_per_category: Number of segments per category
            category_col: Column name for categories
            sampling_rate_key: Key for sampling rate in metadata
            default_sampling_rate: Default if not in metadata
        
        Returns:
            List of TimeSeriesData
        """
        print(f"\nLoading segments from dataset...")
        all_time_series = []
        
        for category in categories:
            print(f"  Category '{category}': ", end="")
            
            # Get segments
            df = dataset.get_random_k_segments_by_category(
                k=k_per_category,
                category_col=category_col,
                category_value=category,
            )
            
            segment_ids = df["segment_id"].tolist()
            print(f"{len(segment_ids)} segments")
            
            # Convert to TimeSeriesData
            for seg_id in segment_ids:
                raw_data = dataset.get_raw(seg_id)
                metadata = dataset.get_metadata(seg_id)
                
                # Convert metadata to ensure no numpy types
                clean_metadata = _convert_numpy_types({
                    **metadata,
                    'category': category,
                    'segment_id': seg_id,
                })
                
                ts = TimeSeriesData(
                    source_id=str(seg_id),
                    data=raw_data,
                    sampling_rate=float(clean_metadata.get(sampling_rate_key, default_sampling_rate)),
                    metadata=clean_metadata,
                )
                all_time_series.append(ts)
        
        print(f"✓ Total loaded: {len(all_time_series)} segments\n")
        return all_time_series
    
    def generate(
        self,
        time_series_list: List[TimeSeriesData],
        plot_configs: List[PlotConfig],
        use_llm_descriptions: bool = True,
        llm_custom_prompt: Optional[str] = None,
        output_filename: str = "example_bank.jsonl",
        verbose: bool = True,
    ) -> str:
        """
        Generate example bank from time series list.
        
        Args:
            time_series_list: List of TimeSeriesData
            plot_configs: List of plot configurations
            use_llm_descriptions: Generate LLM descriptions
            llm_custom_prompt: Custom LLM prompt
            output_filename: Output JSONL filename
            verbose: Print progress
        
        Returns:
            Path to generated example bank JSONL
        """
        if verbose:
            print("="*80)
            print("EXAMPLE BANK GENERATION")
            print("="*80)
            print(f"Input: {len(time_series_list)} time series")
            print(f"Plot types: {[pc.plot_type_label for pc in plot_configs]}")
            print(f"LLM descriptions: {'Yes' if use_llm_descriptions else 'No'}")
            print(f"Embedder: {self.embedding_computer.embedder_type}")
            print("="*80 + "\n")
        
        # Step 1: Generate plots
        if verbose:
            print("Step 1: Generating plots...")
        
        all_examples = []
        for i, ts_data in enumerate(time_series_list, 1):
            if verbose:
                print(f"  [{i}/{len(time_series_list)}] {ts_data.source_id}")
            
            examples = self.plot_generator.generate_all_plots(
                ts_data,
                plot_configs,
                verbose=verbose,
            )
            all_examples.extend(examples)
        
        if verbose:
            print(f"\n✓ Generated {len(all_examples)} plots\n")
        
        # Step 2: LLM descriptions (optional)
        if use_llm_descriptions:
            if verbose:
                print("Step 2: Generating LLM descriptions...")
            
            all_examples = self.llm_describer.describe_batch(
                all_examples,
                custom_prompt=llm_custom_prompt,
                verbose=verbose,
            )
            
            if verbose:
                print(f"✓ Generated descriptions\n")
        
        # Step 3: Compute embeddings
        if verbose:
            print("Step 3: Computing embeddings...")
        
        all_examples = self.embedding_computer.compute_embeddings(
            all_examples,
            compute_text=True,
            compute_vision=True,
            verbose=verbose,
        )
        
        if verbose:
            print(f"✓ Computed embeddings\n")
        
        # Step 4: Save
        if verbose:
            print("Step 4: Saving example bank...")
        
        output_path = self.output_dir / output_filename
        with open(output_path, "w") as f:
            for example in all_examples:
                f.write(json.dumps(example.to_dict()) + "\n")
        
        if verbose:
            print(f"✓ Saved to {output_path}\n")
            self._print_summary(all_examples, output_path)
        
        return str(output_path)
    
    def generate_from_dataset(
        self,
        dataset,
        categories: List[str],
        k_per_category: int,
        plot_configs: Optional[List[PlotConfig]] = None,
        use_llm_descriptions: bool = False,
        output_filename: str = "example_bank.jsonl",
        verbose: bool = True,
    ) -> str:
        """
        Convenience method: load from dataset and generate in one call.
        
        Args:
            dataset: Dataset object
            categories: Categories to sample
            k_per_category: Segments per category
            plot_configs: Plot configurations (uses defaults if None)
            use_llm_descriptions: Generate LLM descriptions
            output_filename: Output filename
            verbose: Print progress
        
        Returns:
            Path to generated example bank JSONL
        """
        # Load data
        time_series_list = self.load_from_dataset(
            dataset,
            categories,
            k_per_category,
        )
        
        # Default plot configs
        if plot_configs is None:
            plot_configs = get_default_plot_configs()
        
        # Generate
        return self.generate(
            time_series_list,
            plot_configs,
            use_llm_descriptions,
            output_filename=output_filename,
            verbose=verbose,
        )
    
    def _print_summary(self, examples: List[ExampleMetadata], output_path: Path):
        """Print generation summary."""
        print("="*80)
        print("GENERATION COMPLETE")
        print("="*80)
        print(f"Total examples: {len(examples)}")
        print(f"Output: {output_path}")
        print(f"Plots: {self.plots_dir}")
        print("="*80)


# ============================================================
# Plot Config Presets
# ============================================================

def get_default_plot_configs(sampling_rate: float = 5000.0) -> List[PlotConfig]:
    """
    Get default plot configurations for neural data.
    
    Includes:
    - Raw time series
    - Spectrogram
    - PSD
    - Power density matrix
    """
    return [
        PlotConfig(
            tool_name="plot_time_series",
            params={
            },
            plot_type_label="raw_timeseries",
        ),
        PlotConfig(
            tool_name="plot_spectrogram",
            params={
            },
            plot_type_label="spectrogram",
        ),
        PlotConfig(
            tool_name="plot_psd",
            params={
            },
            plot_type_label="psd",
        ),
        PlotConfig(
            tool_name="plot_power_density_matrix_hilbert",
            params={
            },
            plot_type_label="power_density_matrix",
        ),
    ]


def get_minimal_plot_configs() -> List[PlotConfig]:
    """Get minimal plot configs (just raw + spectrogram)."""
    return [
        PlotConfig(
            tool_name="plot_time_series",
            params={"channels": None, "figsize": (12, 4)},
            plot_type_label="raw_timeseries",
        ),
        PlotConfig(
            tool_name="plot_spectrogram",
            params={"channels": [0], "n_fft": 256, "figsize": (10, 5)},
            plot_type_label="spectrogram",
        ),
    ]


# ============================================================
# Utility Functions
# ============================================================

def load_example_bank(jsonl_path: str) -> List[ExampleMetadata]:
    """Load example bank from JSONL file."""
    examples = []
    with open(jsonl_path, "r") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                examples.append(ExampleMetadata(**data))
    return examples


def save_example_bank(examples: List[ExampleMetadata], jsonl_path: str):
    """Save example bank to JSONL file."""
    with open(jsonl_path, "w") as f:
        for example in examples:
            f.write(json.dumps(example.to_dict()) + "\n")