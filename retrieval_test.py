import os
import time
import random
import numpy as np
import json
import requests

import dotenv
from collections import defaultdict
import google.generativeai as genai
from openai import OpenAI
from together import Together
import boto3
import anthropic
import seaborn as sns
import matplotlib.pyplot as plt

dotenv.load_dotenv()

GOOGLE_AI_API_KEY = os.getenv("GOOGLE_AI_API_KEY", "")
genai.configure(api_key=GOOGLE_AI_API_KEY)

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "no_api_key"))
together_client = Together(api_key=os.getenv("TOGETHER_API_KEY", "no_api_key"))
anthropic_client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY", "no_api_key")
)


def get_phone_book(n_entries: int) -> list[str]:
    with open("phone_book.txt", "r") as f:
        all_entries = f.readlines()
        # all_entries.reverse()
        # Read a random subset of the phone book
        entries = random.sample(all_entries, n_entries)
    cleaned_entries = [_.strip() for _ in entries]
    return cleaned_entries


def get_system_instructions() -> str:
    inst = ""
    with open("prompt.md", "r") as f:
        inst = f.read()
    return inst


def retrieve_gemini(
    model: str, n_entries: int, n_lookups: int
) -> tuple[str, list[dict], str, float, int]:
    """
    Gets an LLM response for the given user message.
    """
    generation_config = {
        "temperature": 0,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }
    start_time = time.time()
    system_instructions = get_system_instructions()
    phone_book = get_phone_book(n_entries)
    system_instructions += "\n\n# Phone book\n\n" + "\n".join(phone_book)

    selected_records = get_selected_records(phone_book, n_entries, n_lookups)

    model = genai.GenerativeModel(
        model_name=model,
        # model_name="gemini-1.5-flash-002",
        generation_config=generation_config,
        system_instruction=system_instructions,
    )
    user_message = "Please find phone numbers for these names: " + ", ".join(
        [record["record"] for record in selected_records]
    )
    response = model.generate_content(user_message)
    end_time = time.time()
    duration = end_time - start_time
    total_tokens = model.count_tokens(user_message).total_tokens
    print(
        f"Retrieved {n_lookups} / {n_entries} entries in {duration:.2f}s. {total_tokens} tokens."
    )

    matches_raw = response.text.strip().split("\n")
    matches = {}
    for match in matches_raw:
        if " : " in match:
            matches[match.split(" : ")[0].strip()] = match.split(" : ")[1].strip()
        else:
            matches[match] = match

    # Assess retrieval
    result_str, accuracy_data = assess_retrieval(matches, phone_book, selected_records)

    return result_str, accuracy_data, response.text, duration, total_tokens


def get_selected_records(
    phone_book: list[str], n_entries: int, n_lookups: int
) -> list[dict]:
    """Populate selected_records from each bin."""
    bin_size = n_entries // n_lookups
    selected_records = []
    for bin_num in range(n_lookups):
        # Calculate start and end indices for the current bin
        start_idx = bin_num * bin_size
        end_idx = min(start_idx + bin_size, n_entries)

        # Select a random entry within the current bin
        if start_idx < end_idx:  # Ensure the bin is valid
            random_idx = random.randint(start_idx, end_idx - 1)
            record = phone_book[random_idx]
            relative_position = random_idx / n_entries
            selected_records.append(
                {
                    "record": record.split(" : ")[0],
                    "relative_position": relative_position,
                    "bin_num": bin_num,
                }
            )
    return selected_records


def retrieve_openai(
    model: str, n_entries: int, n_lookups: int
) -> tuple[str, list[dict], str, float, int]:
    """
    Gets an LLM response for the given user message.
    """
    start_time = time.time()
    system_instructions = get_system_instructions()
    phone_book = get_phone_book(n_entries)
    system_instructions += "\n\n# Phone book\n\n" + "\n".join(phone_book)

    selected_records = get_selected_records(phone_book, n_entries, n_lookups)

    system_message_role = ""
    if model.startswith("gpt"):
        # For good old gpt4o and earlier models
        system_message_role = "system"
    elif model.startswith("o1-mini"):
        # Specifically for o1-mini since there doesn't seem to be any support for system or developer role
        system_message_role = "user"
    else:
        # For o1 and newer models
        system_message_role = "developer"
    conversation = [{"role": system_message_role, "content": system_instructions}]
    user_message = "Please find phone numbers for these names: " + ", ".join(
        [record["record"] for record in selected_records]
    )
    conversation.append({"role": "user", "content": user_message})

    response = None
    if model.startswith("o1-mini"):
        response = openai_client.chat.completions.create(
            model=model,
            messages=conversation,
            temperature=1,
            # max_tokens=8192,
        )
    else:
        response = openai_client.chat.completions.create(
            model=model,
            messages=conversation,
            temperature=0,
            top_p=0.95,
            # max_tokens=8192,
        )

    response_text = response.choices[0].message.content

    end_time = time.time()
    duration = end_time - start_time
    total_tokens = response.usage.prompt_tokens + response.usage.completion_tokens
    print(
        f"Retrieved {n_lookups} / {n_entries} entries in {duration:.2f}s. {total_tokens} tokens."
    )

    matches_raw = response_text.strip().split("\n")
    matches = {}
    for match in matches_raw:
        if " : " in match:
            matches[match.split(" : ")[0].strip()] = match.split(" : ")[1].strip()
        else:
            matches[match] = match

    # Assess retrieval
    result_str, accuracy_data = assess_retrieval(matches, phone_book, selected_records)

    return result_str, accuracy_data, response_text, duration, total_tokens


def retrieve_together(
    model: str, n_entries: int, n_lookups: int
) -> tuple[str, list[dict], str, float, int]:
    """
    Gets an LLM response for the given user message.
    """
    start_time = time.time()
    system_instructions = get_system_instructions()
    phone_book = get_phone_book(n_entries)
    system_instructions += "\n\n# Phone book\n\n" + "\n".join(phone_book)

    selected_records = get_selected_records(phone_book, n_entries, n_lookups)

    conversation = [{"role": "system", "content": system_instructions}]
    user_message = "Please find phone numbers for these names: " + ", ".join(
        [record["record"] for record in selected_records]
    )
    conversation.append({"role": "user", "content": user_message})

    response = together_client.chat.completions.create(
        model=model,
        messages=conversation,
        temperature=0,
        top_p=0.95,
        stop=["<|eot_id|>", "<|eom_id|>"],
    )

    response_text = response.choices[0].message.content

    end_time = time.time()
    duration = end_time - start_time
    total_tokens = response.usage.prompt_tokens + response.usage.completion_tokens
    print(
        f"Retrieved {n_lookups} / {n_entries} entries in {duration:.2f}s. {total_tokens} tokens."
    )

    print(response_text)

    matches_raw = response_text.strip().split("\n")
    matches = {}
    for match in matches_raw:
        if " : " in match:
            # Split by space, as some models may not follow instructions properly
            # ...and return bullets
            key = match.split(" : ")[0].strip()
            if " " in key:
                key = key.split(" ")[1]

            # Handle possible phone number properly
            # Some models may return xxx-xxx-xxxx instead of (xxx) xxx-xxxx
            # If there are two dashes, format it properly
            value = match.split(" : ")[1].strip()
            if value.count("-") == 2:
                value = f"({value[:3]}) {value[4:7]}-{value[8:]}"
            matches[key] = value
        else:
            # Bad path
            matches[match] = match

    # Assess retrieval
    result_str, accuracy_data = assess_retrieval(matches, phone_book, selected_records)

    return result_str, accuracy_data, response_text, duration, total_tokens


def retrieve_anthropic(
    model: str, n_entries: int, n_lookups: int
) -> tuple[str, list[dict], str, float, int]:
    """
    Gets an LLM response for the given user message, via an Anthropic model
    """
    start_time = time.time()
    system_instructions = get_system_instructions()
    phone_book = get_phone_book(n_entries)
    system_instructions += "\n\n# Phone book\n\n" + "\n".join(phone_book)

    selected_records = get_selected_records(phone_book, n_entries, n_lookups)

    user_message = "Please find phone numbers for these names: " + ", ".join(
        [record["record"] for record in selected_records]
    )

    message = anthropic_client.messages.create(
        model=model,
        max_tokens=4000,
        temperature=0,
        system=system_instructions,
        messages=[
            {"role": "user", "content": [{"type": "text", "text": user_message}]}
        ],
    )

    end_time = time.time()
    duration = end_time - start_time
    total_tokens = message.usage.input_tokens + message.usage.output_tokens
    print(
        f"Retrieved {n_lookups} / {n_entries} entries in {duration:.2f}s. {total_tokens} tokens."
    )

    matches_raw = message.content[0].text.strip().split("\n")
    matches = {}
    for match in matches_raw:
        if " : " in match:
            matches[match.split(" : ")[0].strip()] = match.split(" : ")[1].strip()
        else:
            matches[match] = match

    # Assess retrieval
    result_str, accuracy_data = assess_retrieval(matches, phone_book, selected_records)

    return result_str, accuracy_data, message.content[0].text, duration, total_tokens


def retrieve_amazon(
    model: str, n_entries: int, n_lookups: int
) -> tuple[str, list[dict], str, float, int]:
    """
    Gets an LLM response for the given user message, via an Amazon Bedrock model
    """
    client = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")
    start_time = time.time()
    system_instructions = get_system_instructions()
    phone_book = get_phone_book(n_entries)
    system_instructions += "\n\n# Phone book\n\n" + "\n".join(phone_book)

    selected_records = get_selected_records(phone_book, n_entries, n_lookups)

    user_message = "Please find phone numbers for these names: " + ", ".join(
        [record["record"] for record in selected_records]
    )
    system_message = [{"text": system_instructions}]
    messages = [{"role": "user", "content": [{"text": user_message}]}]
    inf_params = {"temperature": 0.2}

    model_response = client.converse(
        modelId=model,
        messages=messages,
        system=system_message,
        inferenceConfig=inf_params,
    )

    end_time = time.time()
    duration = end_time - start_time
    token_usage = model_response["usage"]
    total_tokens = token_usage["inputTokens"] + token_usage["outputTokens"]
    print(
        f"Retrieved {n_lookups} / {n_entries} entries in {duration:.2f}s. {total_tokens} tokens."
    )

    response_text = model_response["output"]["message"]["content"][0]["text"]
    matches_raw = response_text.strip().split("\n")
    matches = {}
    for match in matches_raw:
        if " : " in match:
            matches[match.split(" : ")[0].strip()] = match.split(" : ")[1].strip()
        else:
            matches[match] = match

    # Assess retrieval
    result_str, accuracy_data = assess_retrieval(matches, phone_book, selected_records)

    return result_str, accuracy_data, response_text, duration, total_tokens


def retrieve_openrouter(
    model: str, n_entries: int, n_lookups: int
) -> tuple[str, list[dict], str, float, int]:
    """
    Gets an LLM response for the given user message, via OpenRouter, using OpenAI's library
    """
    start_time = time.time()
    system_instructions = get_system_instructions()
    phone_book = get_phone_book(n_entries)
    system_instructions += "\n\n# Phone book\n\n" + "\n".join(phone_book)

    selected_records = get_selected_records(phone_book, n_entries, n_lookups)
    user_message = "Please find phone numbers for these names: " + ", ".join(
        [record["record"] for record in selected_records]
    )
    # Not all the models we're testing via openrouter support system instructions
    # So we'll send the system instructions as a user message
    # conversation = [
    #     {"role": "user", "content": system_instructions + "\n\n" + user_message}
    # ]
    conversation = [{"role": "system", "content": system_instructions}]
    # conversation = [{"role": "user", "content": "What is 1+1?"}]
    conversation.append({"role": "user", "content": user_message})

    headers = {
        "Authorization": f"Bearer {os.getenv("OPENROUTER_API_KEY", "no_api_key")}",
    }

    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        data=json.dumps({"model": model, "messages": conversation, "max_tokens": 4000}),
    )
    response_json = response.json()
    print(json.dumps(response_json, indent=4))

    response_text = response_json["choices"][0]["message"]["content"]

    end_time = time.time()
    duration = end_time - start_time
    total_tokens = (
        response_json["usage"]["prompt_tokens"]
        + response_json["usage"]["completion_tokens"]
    )
    print(
        f"Retrieved {n_lookups} / {n_entries} entries in {duration:.2f}s. {total_tokens} tokens."
    )

    matches_raw = response_text.strip().split("\n")
    matches = {}
    for match in matches_raw:
        if " : " in match:
            matches[match.split(" : ")[0].strip()] = match.split(" : ")[1].strip()
        else:
            matches[match] = match

    # Assess retrieval
    result_str, accuracy_data = assess_retrieval(matches, phone_book, selected_records)

    return result_str, accuracy_data, response_text, duration, total_tokens


def assess_retrieval(
    matches: dict, phone_book: list[str], selected_records: list[dict]
) -> tuple[str, list[dict]]:
    phone_book_dict = {
        entry.split(" : ")[0]: entry.split(" : ")[1] for entry in phone_book
    }
    accuracy_data = []

    for name in selected_records:
        bin_num = name["bin_num"]
        record = name["record"]
        correct = record in matches and matches[record] == phone_book_dict.get(record)
        accuracy_data.append(
            {
                "bin_num": bin_num,
                "relative_position": name["relative_position"],
                "correct": correct,
            }
        )

    correct_total = sum(1 for item in accuracy_data if item["correct"])
    accuracy = correct_total / len(accuracy_data) * 100
    return (
        f"Overall Accuracy: {accuracy:.2f}% ({correct_total}/{len(accuracy_data)})",
        accuracy_data,
    )


def run_experiment(model: str, n_entries: int, n_lookups: int, n_runs: int = 10):
    bin_correct_counts = defaultdict(int)
    bin_total_counts = defaultdict(int)

    all_accuracy_data = {}
    raw_responses = {}
    for i in range(n_runs):
        print(f"Run {i + 1} / {n_runs}")
        # Special case for llama 3.1 405b
        if model.startswith("gemini"):
            (
                overall_accuracy_str,
                accuracy_data,
                raw_response,
                duration,
                total_tokens,
            ) = retrieve_gemini(model, n_entries, n_lookups)
        elif model.startswith("gpt") or model.startswith("o1"):
            (
                overall_accuracy_str,
                accuracy_data,
                raw_response,
                duration,
                total_tokens,
            ) = retrieve_openai(model, n_entries, n_lookups)
        elif (
            model.startswith("meta-llama")
            or model.startswith("Qwen")
            or model.startswith("deepseek-ai")
        ):
            (
                overall_accuracy_str,
                accuracy_data,
                raw_response,
                duration,
                total_tokens,
            ) = retrieve_together(model, n_entries, n_lookups)
        elif model.startswith("claude"):
            (
                overall_accuracy_str,
                accuracy_data,
                raw_response,
                duration,
                total_tokens,
            ) = retrieve_anthropic(model, n_entries, n_lookups)
        # elif model.startswith("amazon"):
        #     (
        #         overall_accuracy_str,
        #         accuracy_data,
        #         raw_response,
        #         duration,
        #         total_tokens,
        #     ) = retrieve_amazon(model, n_entries, n_lookups)
        else:
            # Assume OpenRouter
            (
                overall_accuracy_str,
                accuracy_data,
                raw_response,
                duration,
                total_tokens,
            ) = retrieve_openrouter(model, n_entries, n_lookups)

        all_accuracy_data[i + 1] = {
            "overall_accuracy": overall_accuracy_str,
            "accuracy_data": accuracy_data,
            "duration_seconds": f"{duration:.2f}",
            "total_tokens": total_tokens,
        }
        raw_responses[i + 1] = raw_response

        print("\t" + overall_accuracy_str)
        for item in accuracy_data:
            bin_num = item["bin_num"]
            if item["correct"]:
                bin_correct_counts[bin_num] += 1
            bin_total_counts[bin_num] += 1
        # Sleep for 5s to avoid rate limiting
        time.sleep(5)

    # Calculate average accuracy per bin
    average_accuracy = {
        bin_num: bin_correct_counts[bin_num] / bin_total_counts[bin_num]
        for bin_num in range(n_lookups)
    }

    # Save these results, JSONified, to a file
    with open(
        f"results/{model.replace("/", "-").replace(":", "-").replace(".", "-")}-{n_entries}c-{n_lookups}-{n_runs}runs.json",
        "w",
    ) as f:
        f.write(json.dumps(average_accuracy, indent=4))

    # Save raw response to a file
    with open(
        f"raw_responses/{model.replace("/", "-").replace(":", "-").replace(".", "-")}-{n_entries}c-{n_lookups}-{n_runs}runs.txt",
        "w",
    ) as f:
        f.write(json.dumps(raw_responses, indent=4))

    # Save raw accuracy data to a file
    with open(
        f"accuracy_data/{model.replace("/", "-").replace(":", "-").replace(".", "-")}-{n_entries}c-{n_lookups}-{n_runs}runs.json",
        "w",
    ) as f:
        f.write(json.dumps(all_accuracy_data, indent=4))

    # Plot the average accuracy rates
    plot_accuracy_rates(model, average_accuracy, n_entries, n_lookups, n_runs)


def plot_accuracy_rates(
    model: str,
    average_accuracy: dict,
    n_entries: int,
    n_lookups: int,
    n_runs: int,
    degree: int = 2,
):
    # Prepare x and y values for plotting
    x = [i / n_lookups for i in range(n_lookups)]
    y = [average_accuracy.get(i, 0) for i in range(n_lookups)]

    # Fit a polynomial trend line
    poly_coeffs = np.polyfit(x, y, degree)
    poly_func = np.poly1d(poly_coeffs)
    x_smooth = np.linspace(0, 1, 100)
    y_smooth = poly_func(x_smooth)

    # Plot using Seaborn
    sns.set_style(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=x, y=y, color="blue", label="Average Accuracy per Bin")
    sns.lineplot(
        x=x_smooth,
        y=y_smooth,
        color="red",
        label=f"{degree}-Degree Polynomial Trend Line",
    )
    plt.xlabel("Relative Position in Phone Book (0 - 1)")
    plt.ylabel("Average Accuracy Rate")
    plt.ylim(0, 1)
    plt.title(
        f"Retrieval Accuracy (Model: {model}, Haystack Size: {n_entries}, Needles: {n_lookups}, Runs: {n_runs})"
    )
    plt.legend()
    plt.savefig(
        f"charts/{model.replace("/", "-").replace(":", "-").replace(".", "-")}-{n_entries}c-{n_lookups}-{n_runs}runs.png"
    )
    plt.close()


if __name__ == "__main__":
    # Models
    # "gemini-1.5-flash-8b"
    # "gemini-1.5-flash-002"
    # "gemini-1.5-pro-002"
    # "gemini-2.0-flash-exp"
    # "gpt-4-turbo-2024-04-09"
    # "gpt-4o-2024-08-06"
    # "gpt-4o-mini-2024-07-18"
    # "o1-mini-2024-09-12"
    # "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
    # "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
    # "meta-llama/Llama-3.2-3B-Instruct-Turbo"
    # "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo"
    # "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    # "meta-llama/Llama-3.1-405B-Instruct"
    # "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo"
    # "meta-llama/llama-3.1-405b-instruct" -- openrouter
    # "perplexity/llama-3.1-sonar-huge-128k-online"
    # "nousresearch/hermes-3-llama-3.1-405b"
    # "Qwen/Qwen2.5-72B-Instruct-Turbo"
    # "deepseek-ai/DeepSeek-V3"
    # "deepseek/deepseek-r1"

    # Anthropic
    # "claude-3-5-haiku-20241022"
    # "claude-3-5-sonnet-20241022"
    # "claude-3-opus-20240229"

    # OpenRouter models
    # "microsoft/phi-3.5-mini-128k-instruct"
    # "microsoft/phi-3-medium-128k-instruct:free"
    # "ai21/jamba-1-5-large"
    # "ai21/jamba-1-5-mini"
    # "mistralai/ministral-8b"
    # "x-ai/grok-2-1212"
    # "amazon.nova-pro-v1:0"
    # "amazon.nova-lite-v1:0"
    # "amazon.nova-micro-v1:0"
    # via openrouter
    # "amazon/nova-micro-v1"
    # "amazon/nova-lite-v1"
    # "amazon/nova-pro-v1"

    params = [
        # (1000, 10),
        # (1000, 50),
        # (1000, 100),
        # (4000, 10),
        # (4000, 50),
        # (4000, 100),
        # (5000, 10),
        # (5000, 50),
        (5000, 100),
        # (8000, 10),
        # (8000, 50),
        # (8000, 100),
        # (10000, 10),
        # (10000, 50),
        # (10000, 100),
        # (40000, 10),
        # (40000, 100),
        # (80000, 10),
        # (80000, 100),
    ]
    for n_entries, n_lookups in params:
        print(
            f"\n\nRunning experiment for {n_entries} entries and {n_lookups} lookups."
        )
        s_time = time.time()
        run_experiment(
            model="o1-mini-2024-09-12",
            n_entries=n_entries,
            n_lookups=n_lookups,
            n_runs=20,
        )
        print(f"...completed in {time.time() - s_time:.2f}s.")
        time.sleep(5)

    # Run a single experiment
    #
    # run_experiment(
    #     model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    #     n_entries=5000,
    #     n_lookups=10,
    #     n_runs=20,
    # )
