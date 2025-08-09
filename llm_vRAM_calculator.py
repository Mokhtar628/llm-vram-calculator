import argparse

def calculate_vram_requirements(
    model_params_billion: float,
    model_quantization: str,
    max_seq_len: int,
    max_concurrent_requests: int,
    num_layers: int = None,
    kv_attention_heads: int = None,
    head_dim: int = None,
    kv_cache_quantization: str = None,
    overhead_gb: float = 3.0
) -> dict:
    """
    Calculates the estimated VRAM required for serving an LLM.

    Args:
        model_params_billion (float): Number of model parameters in billions (e.g., 27 for 27B).
        model_quantization (str): Quantization precision for model weights (e.g., 'fp32', 'fp16', 'bf16', 'fp8', 'int8', 'int4').
        max_seq_len (int): Maximum sequence length (prompt + generation) for a single request.
        max_concurrent_requests (int): Maximum number of concurrent requests.
        num_layers (int, optional): Number of transformer layers in the model.
                                    Defaults to 48 for 27B models if not provided.
        kv_attention_heads (int, optional): Number of KV attention heads.
                                            Defaults to 8 (for GQA) for 27B models if not provided.
        head_dim (int, optional): Dimensionality of each attention head.
                                  Defaults to 128 for 27B models if not provided.
        kv_cache_quantization (str, optional): Quantization precision for KV cache.
                                               Defaults to `model_quantization` if not provided.
        overhead_gb (float): Estimated general overhead (PyTorch, CUDA context, etc.) in GB.

    Returns:
        dict: A dictionary containing detailed VRAM estimates in GB.
    """

    # --- 1. Define bytes per parameter based on quantization ---
    # This mapping is crucial for accurate memory calculation.
    bytes_per_param_map = {
        'fp32': 4,
        'fp16': 2,
        'bf16': 2,
        'fp8': 1,
        'int8': 1,
        'int4': 0.5,
    }

    model_quantization = model_quantization.lower()
    if model_quantization not in bytes_per_param_map:
        raise ValueError(f"Unsupported model quantization: {model_quantization}. "
                         f"Supported types are: {', '.join(bytes_per_param_map.keys())}")

    # Default KV cache quantization to model quantization if not specified
    if kv_cache_quantization is None:
        kv_cache_quantization = model_quantization
    kv_cache_quantization = kv_cache_quantization.lower()
    if kv_cache_quantization not in bytes_per_param_map:
        raise ValueError(f"Unsupported KV cache quantization: {kv_cache_quantization}. "
                         f"Supported types are: {', '.join(bytes_per_param_map.keys())}")

    bytes_per_param_model = bytes_per_param_map[model_quantization]
    bytes_per_param_kv_cache = bytes_per_param_map[kv_cache_quantization]

    # --- 2. Calculate Model Weights Memory ---
    # Model parameters are in billions, so multiply by 10^9
    model_weights_gb = (model_params_billion * 1_000_000_000 * bytes_per_param_model) / (1024**3)

    # --- 3. Calculate KV Cache Memory ---
    # Use typical values for Gemma 3 27B if not provided
    # These are based on common LLM architectures and Gemma's known GQA
    _num_layers = num_layers if num_layers is not None else 48
    _kv_attention_heads = kv_attention_heads if kv_attention_heads is not None else 8 # GQA
    _head_dim = head_dim if head_dim is not None else 128

    # Total tokens across all concurrent sequences
    max_total_tokens = max_seq_len * max_concurrent_requests

    # KV cache memory formula: 2 (for K and V) * num_layers * num_kv_heads * head_dim * total_tokens * bytes_per_value
    kv_cache_bytes = (
        2 * _num_layers * _kv_attention_heads * _head_dim * max_total_tokens * bytes_per_param_kv_cache
    )
    kv_cache_gb = kv_cache_bytes / (1024**3)

    # --- 4. Total VRAM ---
    total_vram_gb = model_weights_gb + kv_cache_gb + overhead_gb

    return {
        "model_weights_gb": round(model_weights_gb, 2),
        "kv_cache_gb": round(kv_cache_gb, 2),
        "overhead_gb": round(overhead_gb, 2),
        "total_vram_gb": round(total_vram_gb, 2),
        "details": {
            "model_params_billion": model_params_billion,
            "model_quantization": model_quantization,
            "max_seq_len": max_seq_len,
            "max_concurrent_requests": max_concurrent_requests,
            "num_layers": _num_layers,
            "kv_attention_heads": _kv_attention_heads,
            "head_dim": _head_dim,
            "kv_cache_quantization": kv_cache_quantization,
            "max_total_tokens_in_cache": max_total_tokens
        }
    }

def main():
    parser = argparse.ArgumentParser(
        description="Calculate estimated VRAM requirements for LLM serving.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--model_params_billion",
        type=float,
        required=True,
        help="Number of model parameters in billions (e.g., 27 for Gemma 3 27B)."
    )
    parser.add_argument(
        "--model_quantization",
        type=str,
        required=True,
        choices=['fp32', 'fp16', 'bf16', 'fp8', 'int8', 'int4'],
        help="Quantization precision for model weights (e.g., 'fp8')."
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        required=True,
        help="Maximum sequence length (prompt + generation) for a single request."
    )
    parser.add_argument(
        "--max_concurrent_requests",
        type=int,
        required=True,
        help="Maximum number of concurrent requests."
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=48, # Default for Gemma 3 27B
        help="Number of transformer layers in the model."
    )
    parser.add_argument(
        "--kv_attention_heads",
        type=int,
        default=8, # Default for Gemma 3 27B (GQA)
        help="Number of KV attention heads (for Grouped Query Attention)."
    )
    parser.add_argument(
        "--head_dim",
        type=int,
        default=128, # Default for Gemma 3 27B
        help="Dimensionality of each attention head."
    )
    parser.add_argument(
        "--kv_cache_quantization",
        type=str,
        choices=['fp32', 'fp16', 'bf16', 'fp8', 'int8', 'int4'],
        help="Quantization precision for KV cache (defaults to model_quantization if not specified)."
    )
    parser.add_argument(
        "--overhead_gb",
        type=float,
        default=3.0,
        help="Estimated general overhead (PyTorch, CUDA context, etc.) in GB."
    )

    args = parser.parse_args()

    try:
        results = calculate_vram_requirements(
            model_params_billion=args.model_params_billion,
            model_quantization=args.model_quantization,
            max_seq_len=args.max_seq_len,
            max_concurrent_requests=args.max_concurrent_requests,
            num_layers=args.num_layers,
            kv_attention_heads=args.kv_attention_heads,
            head_dim=args.head_dim,
            kv_cache_quantization=args.kv_cache_quantization,
            overhead_gb=args.overhead_gb
        )

        print("\n--- LLM VRAM Requirement Estimation ---")
        print(f"Model Parameters: {results['details']['model_params_billion']} Billion")
        print(f"Model Quantization: {results['details']['model_quantization'].upper()}")
        print(f"KV Cache Quantization: {results['details']['kv_cache_quantization'].upper()}")
        print(f"Max Sequence Length: {results['details']['max_seq_len']}")
        print(f"Max Concurrent Requests: {results['details']['max_concurrent_requests']}")
        print(f"Number of Layers: {results['details']['num_layers']}")
        print(f"KV Attention Heads: {results['details']['kv_attention_heads']}")
        print(f"Head Dimension: {results['details']['head_dim']}")
        print(f"Max Total Tokens in KV Cache: {results['details']['max_total_tokens_in_cache']}\n")

        print(f"1. Model Weights Memory: {results['model_weights_gb']} GB")
        print(f"2. KV Cache Memory:     {results['kv_cache_gb']} GB")
        print(f"3. Other Overheads:     {results['overhead_gb']} GB (estimated)")
        print(f"----------------------------------------")
        print(f"Total Estimated VRAM:   {results['total_vram_gb']} GB")
        print("----------------------------------------")
        print("\nNote: This is an estimation. Actual VRAM usage may vary.")
        print("Consider adding a buffer for robust serving.")

    except ValueError as e:
        print(f"Error: {e}")
        parser.print_help()

if __name__ == "__main__":
    main()


"""
    python llm_vRAM_calculator.py --model_params_billion 27 --model_quantization fp8 --max_seq_len 2048 --max_concurrent_requests 32 --num_layers 48 --kv_attention_heads 8 --head_dim 128 --kv_cache_quantization fp8 --overhead_gb 3.0

    **Explanation of Arguments:**
    * `--model_params_billion`: The number of parameters in billions (e.g., `27` for 27B).
    * `--model_quantization`: The precision of the model weights (`fp32`, `fp16`, `bf16`, `fp8`, `int8`, `int4`).
    * `--max_seq_len`: The maximum total sequence length (prompt + generated tokens) for a single request.
    * `--max_concurrent_requests`: The maximum number of requests that will be processed simultaneously.
    * `--num_layers`: (Optional) Number of transformer layers. Defaults to 48 for Gemma 3 27B.
    * `--kv_attention_heads`: (Optional) Number of Key-Value attention heads. Defaults to 8 for Gemma 3 27B (due to Grouped Query Attention).
    * `--head_dim`: (Optional) The dimensionality of each attention head. Defaults to 128 for Gemma 3 27B.
    * `--kv_cache_quantization`: (Optional) The precision for the KV cache. If not specified, it defaults to the `model_quantization`.
    * `--overhead_gb`: (Optional) An estimated fixed overhead for other components (like PyTorch, CUDA context, Encoders, etc.). Defaults to 3.0 GB.

"""