# Ollama Performance Measurement Tool Documentation

## Conceptual Overview

A tool for measuring the performance of Ollama LLM inference. It captures various metrics such as CPU usage, RAM consumption, token speed, and stores them in CSV files for later analysis. This is specialized on architectures running on apple silicon.

## Core Measurement Concepts

### Three-Tiered Metrics Approach

The tool implements a three-tiered measurement methodology to provide comprehensive insights:

1. **System Resource Metrics** (ChatMetrics)
   - Captures the hardware resource utilization at regular intervals
   - Records CPU, RAM, and swap memory consumptionpwd
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

## Get Started

### Prerequisites for macOS

- Python 3.8 or higher
- Ollama with an installed model (default: llama3.2)
- Pip for installing dependencies

#### Installing Python on macOS

On a fresh macOS system:

1. **Install Homebrew** (if not already installed):
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```
   Follow the instructions to set up Homebrew.

2. **Install required build dependencies**:
   ```bash
   brew install cmake pkg-config
   ```
   These are needed for building some of the Python dependencies.

3. **Install Python**:
   ```bash
   brew install python
   ```
   This installs Python 3 and pip.

4. **Verify the installation**:
   ```bash
   python3 --version
   pip3 --version
   ```

### Installation

#### Installing Ollama

For macOS, the recommended way is to install Ollama via Homebrew:

```bash
# Install Ollama
brew install ollama

# Start the Ollama server
ollama serve
```

Note: The `ollama serve` command will start the Ollama server and block the current terminal. You should keep this terminal window open while using Ollama.

#### Running Ollama in a separate terminal

For a better workflow, you'll need at least two terminal sessions:

1. **Terminal 1**: Run the Ollama server
   ```bash
   ollama serve
   ```

2. **Terminal 2**: Pull the model and run your performance tool
   ```bash
   # First, pull the model you want to use
   ollama pull llama3.2
   
   # Then navigate to your project directory and run the tool
   cd /path/to/project
   python main.py
   ```

#### Setting up the project

1. Clone the repository:
   ```bash
   git clone https://github.com/PeterFromBitworkers/ollama-performance-tool.git
   cd ollama-performance-tool
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Configuration

Configuration is done via the `Config` class in `main.py`. Here are some important settings:

- `MODEL`: The Ollama model to use (default: "llama3.2")
- `CTX`: Context size for the model (default: 2048)
- `SYSTEM_MEASUREMENT_INTERVAL`: Interval for system metrics (default: every 20 tokens)
- `MODEL_FAMILY`: HuggingFace model for token counting

### Usage

Start the tool with:

```bash
python main.py
```

A chat interface will open where you can interact with the model. During interaction, metrics are captured and stored in CSV files in the `metrics` folder:

- `system.csv`: System metrics (CPU, RAM)
- `message_results.csv`: Message metrics (token count, speed)
- `tokens.csv`: Token generation metrics

End the chat by typing "exit", "quit", or "q".

### Analyzing Results

The generated CSV files can be analyzed using tools like Python Pandas, Excel, or Grafana to gain performance insights.