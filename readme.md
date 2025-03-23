# Ollama Performance Measurement Tool Documentation

## Conceptual Overview

The Ollama Performance Measurement Tool is a specialized CLI application designed to capture and analyze the performance characteristics of Large Language Models (LLMs) during local inference. This tool addresses the critical need for quantitative measurement of LLM performance, particularly on Apple Silicon architectures with shared memory constraints.

## Core Measurement Concepts

### Three-Tiered Metrics Approach

The tool implements a three-tiered measurement methodology to provide comprehensive insights:

1. **System Resource Metrics** (ChatMetrics)
   - Captures the hardware resource utilization at regular intervals
   - Records CPU, RAM, and swap memory consumption
   - Provides insights into the computational demands of inference

2. **Token-Level Metrics** (TokenMetrics)
   - Measures the generation time for each individual token
   - Creates a precise temporal map of the inference process
   - Enables identification of performance bottlenecks at the token level

3. **Message-Level Metrics** (MessageMetrics)
   - Aggregates statistics for complete message exchanges
   - Calculates key performance indicators like Time To First Token (TTFT) and Tokens Per Second (TPS)
   - Establishes baseline performance metrics for comparative analysis

### Temporal Measurement Points

The tool establishes specific temporal markers to segment the inference process:
- `time_user_entered_input`: When the user completes their prompt
- `time_first_token_inferenced`: When the first token response appears
- Streaming token generation: Continuous monitoring throughout response generation

This temporal framework allows for detailed analysis of latency patterns and throughput characteristics across the entire inference lifecycle.

## Configuration Philosophy

The configuration system is designed around flexibility and reproducibility:

- **Model Specification**: Supports different Ollama-compatible models, recognizing that performance varies significantly across model architectures
- **Context Window Manipulation**: Allows testing with different context sizes to understand memory-performance tradeoffs
- **Measurement Granularity**: Configurable system measurement intervals balance accuracy against measurement overhead
- **Session Identification**: Unique chat IDs enable longitudinal analysis across multiple inference sessions

Please Note the Measurment Granularity is introducing a very lil portion on unprciseness for the time elapsing when the measurement is taking place, so we think a value of every 20 token might be reasonable. To monitor the uncertanty the lost time is still printed out to the cli however is not effective in the files.