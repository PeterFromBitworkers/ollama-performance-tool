import ollama
import os
import time
import psutil
import datetime
from dataclasses import dataclass
from typing import List, Dict, Any
from transformers import AutoTokenizer

# work on readme
# fill requirments.txt
# git

# Configuration as a simple class instead of globals
@dataclass
class Config:
    CHAT_FILE = "system.csv"
    MESSAGE_FILE = "message_results.csv"
    TOKEN_FILE = "tokens.csv"
    OVERWRITE_FILES: bool = False
    MODEL: str = "llama3.2"
    CTX: str = 2048
    SYSTEM_MEASUREMENT_INTERVAL: int = 20
    
    # Needed for the tokenizer ( a hf library that is used to count the tokens)
    #  openlm-research/open_llama_3b
    #  microsoft/phi-1_5
    #  google/gemma-2b
    #  mistralai/Mistral-7B-v0.1	
    MODEL_FAMILY: str = "openlm-research/open_llama_3b"

# Global config instance
config = Config()

@dataclass
class ChatMetrics:
    timestamp: float
    id: int
    message_id: int
    chat_id: str
    token_id: int
    system_cpu_percent: float
    system_ram_total_gb: float
    system_ram_available_gb: float
    system_ram_used_gb: float
    system_ram_percent: float
    system_swap_used: float
    type: str = "system"  # For internal processing
    lost_time: float = 0

    
    # Class attribute for headers
    headers = [
        "TIMESTAMP", "ID", "MESSAGE ID", "CHAT ID", "TOKEN ID", 
        "CPU (%)", "SYSTEM RAM (GB)", "AVAIL RAM (GB)", 
        "USED RAM (GB)", "USED RAM (%)", "SWAP (GB)", "DURATION (s)"
    ]
    
    def to_csv_row(self) -> str:
        """Convert the metrics to a CSV row."""
        values = [
            self.timestamp, self.id, self.message_id, self.chat_id, self.token_id,
            self.system_cpu_percent,
            self.system_ram_total_gb, self.system_ram_available_gb,
            self.system_ram_used_gb, self.system_ram_percent, self.system_swap_used,
            self.lost_time
        ]
        return ",".join(str(value) for value in values)

@dataclass
class MessageMetrics:
    timestamp: float
    id: int
    chat_id: str
    model: str
    ctx_setting: int
    in_tokens: int
    out_tokens: int
    total_tokens: int
    idle_time_prior_message: float
    tokens_per_second: float
    avg_system_cpu_percent: float
    avg_system_ram: float 
    type: str = "message"  # For internal processing

    
    # Class attribute for headers
    headers = [
        "TIMESTAMP", "ID", "CHAT ID", "MODEL", "CTX SETTING",
        "IN TOKEN","OUT TOKEN", "TOTAL TOKEN", 
        "TTFT", "TPS", "CPU (%)", "RAM (gb)"
    ]
    
    def to_csv_row(self) -> str:
        """Convert the metrics to a CSV row."""
        values = [
            self.timestamp, self.id, self.chat_id, self.model, self.ctx_setting,
            self.in_tokens, self.out_tokens, self.total_tokens,
            self.idle_time_prior_message, self.tokens_per_second,
            self.avg_system_cpu_percent, self.avg_system_ram
        ]
        return ",".join(str(value) for value in values)

@dataclass
class TokenMetrics:
    timestamp: float
    id: int
    message_id: int
    chat_id: str
    elapsed_time: float
    type: str = "token"  # For internal processing
    
    # Class attribute for headers
    headers = [
        "TIMESTAMP", "ID", "MESSAGE ID", "CHAT ID", 
        "ELAPSED TIME (s)"
    ]
    
    def to_csv_row(self) -> str:
        """Convert the metrics to a CSV row."""
        values = [
            self.timestamp, self.id, self.message_id, self.chat_id,
            self.elapsed_time
        ]
        return ",".join(str(value) for value in values)

def get_resource_metrics(id: int, message_id: int, chat_id: str, token_id: int) -> ChatMetrics:
    """
    Captures detailed system resource metrics.
    Returns a ChatMetrics object with the actual metrics.
    """
    timestamp = time.time()
    
    # System memory information
    system_memory = psutil.virtual_memory()
    swap_info = psutil.swap_memory()
    
    # System CPU !!! NON BLOCKING !!!
    system_cpu_percent = psutil.cpu_percent(interval=None)

    # Look at the watch
    time_end_of_measurement = time.time()
    
    return ChatMetrics(
        timestamp=timestamp,
        id=id,
        message_id=message_id,
        chat_id=chat_id,
        token_id=token_id,
        system_cpu_percent=system_cpu_percent, #CPU usage in percent
        system_ram_total_gb=system_memory.total / (1024**3), # Total RAM in GB always 16 here
        system_ram_available_gb=system_memory.available / (1024**3), # Available RAM in GB (not used by the system)
        system_ram_used_gb=system_memory.used / (1024**3), #currently used RAM in GB
        system_ram_percent=system_memory.percent, #RAM usage in percent
        system_swap_used=swap_info.used / (1024**3), # Swap used in GB
        lost_time = time_end_of_measurement - timestamp
    )

def get_inference_metrics(  message_token_count: int,
                            message_id: int, 
                            chat_id: str, 
                            time_user_entered_input) -> TokenMetrics:
    """
    Captures inference metrics for token generation.
    """
    return TokenMetrics(
        timestamp=time.time(),
        id=message_token_count,
        message_id=message_id,
        chat_id=chat_id,
        elapsed_time=time.time() - time_user_entered_input,
    )

def get_message_metrics(message_id: int, 
                        chat_id: str, 
                        time_user_entered_input,
                        time_first_token_inferenced, 
                        chat_messages, 
                        all_metrics) -> MessageMetrics:
    """
    Calculates metrics for a message.
    Returns a MessageMetrics object with the metrics.
    """

    in_tokens = 0
    out_tokens = 0 # this is the actual inference performance in this iteration
    in_messages = ""
    out_messages = ""

    # Alle bis auf die letzte message sind in messages
    # da ja auch die inferenzen der vorherigen iterationen
    # jetzt input sind.

    # iterate over all messages but the last one
    for message in chat_messages[:-1]:
        in_messages += message["content"]

    out_messages = chat_messages[-1]["content"]

    in_tokens = count_tokens(in_messages)
    out_tokens = count_tokens(out_messages)
    total_tokens = in_tokens + out_tokens

    # idle time, token per second, 
    # sum the lost time of all tokens
    sum_lost_time = 0
    for m in all_metrics:
        if m.type == "system" and m.message_id == message_id:
            sum_lost_time += m.lost_time

    time_elapsed_message = time.time() - time_user_entered_input
    time_prior_first_token = time_first_token_inferenced - time_user_entered_input # from when the user hits enter until the first token is printed
    # ignoring lost time here as measurements are non blocking meanwhile
    # however printing the losts below in cas they will ever be significant
    tokens_per_second = out_tokens / time_elapsed_message

    print(f"elapsed time: {time_elapsed_message:.2f}")
    print(f"lost time: {sum_lost_time}")
    

    print(f"in_token: {in_tokens}")
    print(f"out_token: {out_tokens}")
    print(f"total_token: {total_tokens}")

    print(f"ttft: {time_prior_first_token:.2f}")
    print(f"tps: {tokens_per_second:.2f}")
    
    # Calculate averages
    system_metrics = [m for m in all_metrics if m.type == "system"]
    
    if system_metrics:
        avg_system_cpu_percent = sum(m.system_cpu_percent for m in system_metrics) / len(system_metrics)
        avg_system_ram = sum(m.system_ram_used_gb for m in system_metrics) / len(system_metrics)
    else:
        avg_system_cpu_percent = 0
        avg_system_ram = 0

    print(f"avg cpu: {avg_system_cpu_percent:.2f}")
    print(f"avg ram: {avg_system_ram:.2f}")
    
    return MessageMetrics(
        timestamp=time.time(),
        id=message_id,
        chat_id=chat_id,
        model=config.MODEL,
        ctx_setting=config.CTX,
        in_tokens=in_tokens,
        out_tokens=out_tokens,
        total_tokens=total_tokens,
        idle_time_prior_message=time_prior_first_token,
        tokens_per_second=tokens_per_second,
        avg_system_cpu_percent=avg_system_cpu_percent,
        avg_system_ram=avg_system_ram
    )

def simple_chat():
    """
    Main chat function that interacts with the LLM and collects metrics.
    """

    model_name = config.MODEL
  
    # Initialize the chat session
    chat_messages = []  # actual chat messages as by model api
    system_metrics_counter = 0
    
    print("-" * 40)
    print(f"Chatting with {model_name} (type 'exit' to quit)")
    print("-" * 40)
    
    while True:
        
        user_input = input("\nYou: ")
        time_user_entered_input = time.time() # user hits enter after typing the message

        # need ONE for the whole chat (not per message)
        if len(chat_messages) == 0:
            chat_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S") + user_input[:10]

        if user_input.lower() in ["exit", "quit", "q"]:
            break
            
        chat_messages.append({"role": "user", "content": user_input})
        response = ollama.chat(
            model=model_name,
            messages=chat_messages,
            stream=True,
            options={"num_ctx": config.CTX}
        )
        
        print("\nAssistant: ", end="")
        assistant_message = ""
        
        # Measurements for this message
        message_token_count = 0  # the amount of tokens in this message
        all_metrics = []
        message_id = len(chat_messages) // 2 + 1
        

        for chunk in response:
            # take the time if this is the first chunk
            if message_token_count == 0:
                time_first_token_inferenced = time.time()

            content = chunk.get('message', {}).get('content', '')
            if content:
                print(content, end="", flush=True)
                assistant_message += content

                # Measure token generation
                message_token_count += 1
                time_this_token_creation = time.time()
                
                # System metrics at regular intervals
                if message_token_count == 1 or message_token_count % config.SYSTEM_MEASUREMENT_INTERVAL == 0:
                    system_metrics_counter += 1
                    system_metrics = get_resource_metrics(
                        system_metrics_counter, 
                        message_id, 
                        chat_id, 
                        message_token_count
                    )
                    all_metrics.append(system_metrics)
                
                # Token metrics for each token
                token_metrics = get_inference_metrics(
                    message_token_count, 
                    message_id, 
                    chat_id,
                    time_user_entered_input
                )
                all_metrics.append(token_metrics)

        print("\n..calc statistics......................................")  # Newline after response
        chat_messages.append({"role": "assistant", "content": assistant_message})


        
        # LLM is done printing and internaly saving the message
        # Calculate metrics for this message
        message_metrics = get_message_metrics(
            message_id, 
            chat_id, 
            time_user_entered_input,
            time_first_token_inferenced,
            chat_messages, 
            all_metrics
        )
        all_metrics.append(message_metrics)

        # Output statistics
        write_statistics(all_metrics)

        print("..done......................................")

def write_statistics(metrics):
    """
    Writes metrics to CSV files based on their type.
    Each metric type goes to its corresponding file.
    
    Args:
        metrics: List of metric objects
    """
    # Define file paths within the 'metrics' folder
    chat_file_path = os.path.join("metrics", config.CHAT_FILE)
    message_file_path = os.path.join("metrics", config.MESSAGE_FILE)
    token_file_path = os.path.join("metrics", config.TOKEN_FILE)
    
    # Open files for appending
    with open(chat_file_path, "a") as system_file, \
         open(message_file_path, "a") as message_file, \
         open(token_file_path, "a") as token_file:
        
        # Write data by type
        for metric in metrics:
            if metric.type == "system":
                system_file.write(metric.to_csv_row() + "\n")
            elif metric.type == "message":
                message_file.write(metric.to_csv_row() + "\n")
            elif metric.type == "token":
                token_file.write(metric.to_csv_row() + "\n")

    print(f"Statistics written to {chat_file_path}, {message_file_path}, and {token_file_path}")

def initialize_files(delete_existing=True):
    """
    Initialize CSV files for metrics collection.
    
    Args:
        delete_existing (bool): Whether to delete existing CSV files. Default is True.
    """
    # Ensure the 'metrics' directory exists
    if not os.path.exists("metrics"):
        os.makedirs("metrics")
        print("Created 'metrics' directory")
    
    # Define file configurations mapping
    files_config = {
        os.path.join("metrics", config.CHAT_FILE): ChatMetrics.headers,
        os.path.join("metrics", config.MESSAGE_FILE): MessageMetrics.headers,
        os.path.join("metrics", config.TOKEN_FILE): TokenMetrics.headers
    }
    
    # Delete existing files if requested
    if config.OVERWRITE_FILES:
        if delete_existing:
            for filepath in files_config:
                if os.path.exists(filepath):
                    os.remove(filepath)
                    print(f"Deleted existing {filepath}")
    
    # Create files and write headers
    for filepath, headers in files_config.items():
        if not os.path.exists(filepath):
            with open(filepath, "w") as file:
                file.write(",".join(headers) + "\n")
                print(f"Created {filepath} with headers")
        else:
            print(f"{filepath} already exists, headers not modified")

def count_tokens(text):
    # Cache tokenizer to avoid reloading
    if not hasattr(count_tokens, "tokenizer"):
        count_tokens.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_FAMILY)
    
    tokens = count_tokens.tokenizer.encode(text)
    return len(tokens)

def main():
    """
    Main function to run the program.
    """

    initialize_files()  # Initialize files with headers
    simple_chat()  # Start the chat session


if __name__ == "__main__":
    main()