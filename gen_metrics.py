import argparse
import numpy as np
import matplotlib.pyplot as plt
import json


def get_retrieval_accuracy_stats(
    results: dict, draw: bool = False
) -> tuple[float, float, float, float, float]:
    values = list(results.values())

    # Generate x values corresponding to the indices of the values
    x = np.arange(len(values))

    # Fit a 3rd-degree polynomial to the data
    coefficients = np.polyfit(x, values, 3)

    # Create a polynomial function from the coefficients
    polynomial = np.poly1d(coefficients)

    # Generate a smooth range of x values for plotting and finding the minimum
    x_smooth = np.linspace(0, len(values) - 1, 1000)
    y_smooth = polynomial(x_smooth)

    # Find the minimum value of the best-fit line
    # Cap it at 0 (if <0) and 1 (if >1)
    min_y = max(np.min(y_smooth), 0)
    min_x = x_smooth[np.argmin(y_smooth)]
    max_y = min(np.max(y_smooth), 1)
    max_x = x_smooth[np.argmax(y_smooth)]
    avg_y = np.mean(y_smooth)

    if draw:
        # Plot the original values and the best-fit line
        plt.scatter(x, values, label="Original Values", color="blue", s=10)
        plt.plot(x_smooth, y_smooth, label="3rd-Degree Polynomial Fit", color="red")
        plt.axvline(
            min_x, color="green", linestyle="--", label=f"Minimum at x={min_x:.2f}"
        )
        plt.axhline(
            min_y, color="orange", linestyle="--", label=f"Minimum Value={min_y:.2f}"
        )
        plt.legend()
        plt.title("3rd-Degree Polynomial Best Fit")
        plt.xlabel("Index")
        plt.ylabel("Values")
        plt.ylim(0, 1)
        plt.show()

    return min_x, min_y, max_x, max_y, avg_y


def calc_metrics(model_name: str, max_haystack_size: int):
    print(f"\nCalculating metrics for {model_name} - haystack={max_haystack_size}")

    with open(f"results/{model_name}-{max_haystack_size}c-100-20runs.json", "r") as f:
        raw_data_max_c = f.read()
    with open(f"results/{model_name}-{max_haystack_size}c-10-20runs.json", "r") as f:
        raw_data_max_c_10_n = f.read()
    with open(f"results/{model_name}-1000c-100-20runs.json", "r") as f:
        raw_data_1000_c = f.read()

    results_max_c = json.loads(raw_data_max_c)
    results_max_c_10_n = json.loads(raw_data_max_c_10_n)
    results_1000_c = json.loads(raw_data_1000_c)
    min_x_max_c, min_y_max_c, max_x_max_c, max_y_max_c, avg_y_max_c = (
        get_retrieval_accuracy_stats(results_max_c)
    )
    (
        min_x_max_c_10_n,
        min_y_max_c_10_n,
        max_x_max_c_10_n,
        max_y_max_c_10_n,
        avg_y_max_c_10_n,
    ) = get_retrieval_accuracy_stats(results_max_c_10_n)
    min_x_1000_c, min_y_1000_c, max_x_1000_c, max_y_1000_c, avg_y_1000_c = (
        get_retrieval_accuracy_stats(results_1000_c)
    )

    print(f"\nH={max_haystack_size:,}, N=100:")
    print(f"\tAccuracy_max = {max_y_max_c:.3f} @ x={max_x_max_c:.3f}.")
    print(f"\tAccuracy_min = {min_y_max_c:.3f} @ x={min_x_max_c:.3f}.")
    print(f"\tAccuracy_avg = {avg_y_max_c:.3f}")

    print(f"\nH={max_haystack_size:,}, N=10:")
    print(f"\tAccuracy_max = {max_y_max_c_10_n:.3f} @ x={max_x_max_c_10_n:.3f}.")
    print(f"\tAccuracy_min = {min_y_max_c_10_n:.3f} @ x={min_x_max_c_10_n:.3f}.")
    print(f"\tAccuracy_avg = {avg_y_max_c_10_n:.3f}")

    print("\nH=1,000, N=100:")
    print(f"\tAccuracy_max = {max_y_1000_c:.3f} @ x={max_x_1000_c:.3f}.")
    print(f"\tAccuracy_min = {min_y_1000_c:.3f} @ x={min_x_1000_c:.3f}.")
    print(f"\tAccuracy_avg = {avg_y_1000_c:.3f}")

    # Calculate: Context Saturation Index (CSI)
    # How much does retrieval accuracy drop (as a %) when going from M=1K to the specified haystack size,
    # for 100 needles?
    # Scale of 0% to 100%, lower is better.
    csi = max(min_y_1000_c - min_y_max_c, 0) / min_y_1000_c if min_y_1000_c > 0 else 1

    # Calculate: Attention Decay Index (ADI)
    # How much does retrieval accuracy drop (as a %) from (typically) the beginning of the context
    # to later in the context (typically near the end) when retrieval accuracy is the lowest,
    # for 100 needles and the specified haystack size
    # Scale of 0% to 100%, lower is better.
    adi = max(max_y_max_c - min_y_max_c, 0) / max_y_max_c if max_y_max_c > 0 else 1

    # Calculate: Needle Overload Index (NOI)
    # How much does retrieval accuracy drop (as a %) when we
    # increase the no. of needles from 10 to 100, for the specified haystack size?
    # Scale of 0% to 100%, lower is better.
    noi = (
        max(min_y_max_c_10_n - min_y_max_c, 0) / min_y_max_c_10_n
        if min_y_max_c_10_n > 0
        else 1
    )

    # Calculate: Retrieval Resilience Index (RRI)
    # 100% minus the average of CSI, ADI and NOI.
    # Scale of 0% to 100%.
    # A higher score indicates greater resilience to context saturation, attention decay and needle overload.
    rri = 1 - (csi + adi + noi) / 3.0

    # Calculate: Long Context Retrieval Dependability Score (LCRDS)
    # Average retrieval accuracy percentage at the specified haystack size for 100 needles,
    # discounted by RRI as a percentage.
    # Scale of 0 to 100, higher is better.
    lcrds = avg_y_max_c * 100.0 * rri

    print("\n----- METRICS -----\n")
    print(f"CSI = {csi:.3f}")
    print(f"ADI = {adi:.3f}")
    print(f"NOI = {noi:.3f}")
    print(f"RRI = {rri:.3f}")
    print(f"\n* LCRDS = {lcrds:.2f}%\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate retrieval accuracy metrics for a given model and haystack size."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Full model name to calculate metrics for.",
    )
    parser.add_argument(
        "--haystack",
        type=int,
        required=True,
        help="The maximum haystack size (no. of phone book entries).",
    )
    args = parser.parse_args()

    # Supported models
    supported_models = [
        # Gemini models
        "gemini-1.5-flash-8b",
        "gemini-1.5-flash-002",
        "gemini-1.5-pro-002",
        "gemini-2.0-flash-exp",
        # OpenAI models
        "gpt-4-turbo-2024-04-09",
        "gpt-4o-2024-08-06",
        "gpt-4o-mini-2024-07-18",
        # Llama models
        "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        "meta-llama/Llama-3.2-3B-Instruct-Turbo",
        "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
        "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "meta-llama/Llama-3.1-405B-Instruct",
        "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
        "meta-llama/llama-3.1-405b-instruct",
        "perplexity/llama-3.1-sonar-huge-128k-online",
        "nousresearch/hermes-3-llama-3.1-405b",
        # Anthropic
        "claude-3-5-haiku-20241022",
        "claude-3-5-sonnet-20241022",
        "claude-3-opus-20240229",
        # Other models
        "Qwen/Qwen2.5-72B-Instruct-Turbo",
        "deepseek-ai/DeepSeek-V3",
        # via OpenRouter
        "microsoft/phi-3.5-mini-128k-instruct",
        "microsoft/phi-3-medium-128k-instruct:free",
        "ai21/jamba-1-5-large",
        "ai21/jamba-1-5-mini",
        "mistralai/ministral-8b",
        "x-ai/grok-2-1212",
        "amazon.nova-pro-v1:0",
        "amazon.nova-lite-v1:0",
        "amazon.nova-micro-v1:0",
        "amazon/nova-micro-v1",
        "amazon/nova-lite-v1",
        "amazon/nova-pro-v1",
    ]

    if args.model not in supported_models:
        print(f"Model {args.model} is not supported.")
        exit(1)

    model_name = args.model.replace("/", "-").replace(":", "-").replace(".", "-")
    calc_metrics(model_name=model_name, max_haystack_size=args.haystack)
