import json
import glob
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
import os


def draw_chart(
    models: list[str], haystack_size: int, needles: int, file_prefix: str = ""
):
    """
    Draws a chart comparing the performance of various models for the retrieval task.
    Saves the chart to agg_charts/{haystack_size}c{needles}.png
    """
    # Prepare a DataFrame to store all model data for plotting
    plot_data = pd.DataFrame()

    # Load data for each model from respective JSON files
    for model in models:
        filename = f"results/{model}-{haystack_size}c-{needles}-20runs.json"

        # Load JSON data
        try:
            with open(filename, "r") as file:
                data = json.load(file)
        except FileNotFoundError:
            print(f"File not found: {filename}")
            continue

        # Convert data into a DataFrame format suitable for plotting, with score as %
        model_df = pd.DataFrame(
            {
                "Relative Position in Phone Book": [
                    int(k) / (len(data) - 1) for k in data.keys()
                ],
                "Score (%)": [v * 100 for v in data.values()],
                "Model": model,
            }
        )

        # Append to the main DataFrame
        plot_data = pd.concat([plot_data, model_df], ignore_index=True)

    # Set up the plot with brighter color palette and enhanced aesthetics
    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("bright")

    plt.figure(figsize=(12, 8))

    # Scatter plot with lower opacity
    sns.scatterplot(
        data=plot_data,
        x="Relative Position in Phone Book",
        y="Score (%)",
        hue="Model",
        palette=palette,
        alpha=0.3,
        legend=False,
    )

    # Polynomial regression plots for each model
    for model in plot_data["Model"].unique():
        model_data = plot_data[plot_data["Model"] == model]
        x = model_data["Relative Position in Phone Book"]
        y = model_data["Score (%)"]

        # Fit 2-degree polynomial
        poly_2 = np.polyfit(x, y, 2)
        y_pred_2 = np.polyval(poly_2, x)
        mse_2 = mean_squared_error(y, y_pred_2)

        # Fit 3-degree polynomial
        poly_3 = np.polyfit(x, y, 3)
        y_pred_3 = np.polyval(poly_3, x)
        mse_3 = mean_squared_error(y, y_pred_3)

        # Choose the better fit
        if mse_2 < mse_3:
            better_fit = 2
            best_poly = poly_2
        else:
            better_fit = 3
            best_poly = poly_3

        # Plot the better fit
        x_fit = np.linspace(x.min(), x.max(), 500)
        y_fit = np.polyval(best_poly, x_fit)

        plt.plot(
            x_fit,
            y_fit,
            label=f"{model} (Best Fit: {better_fit}-degree)",
            linewidth=2.5,
        )

    plt.title(
        f"Model Performance: phone book size M={haystack_size}, retrieving N={needles} numbers",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    plt.xlabel("Relative Position in Phone Book (0 - 1)")
    plt.ylabel("Score (%)")
    plt.xlim(0, 1)
    plt.ylim(0, 100)

    # Place legend below the plot, outside the plot area, arranged horizontally
    plt.legend(
        title="Model",
        loc="upper center",
        bbox_to_anchor=(
            0.5,
            -0.15,
        ),  # Position the legend outside the plot at the bottom
        ncol=3,  # Arrange legend items in a row (adjust ncol as needed)
        frameon=False,  # Optional: Remove the box around the legend
    )
    # Adjust layout to make room for the legend outside the plot
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save the plot
    output_dir = "agg_charts"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/{file_prefix}{'_' if file_prefix else ''}{haystack_size}c{needles}.png"
    plt.savefig(output_file, bbox_inches="tight")
    plt.close()
    print(f"Chart saved as {output_file}")


def draw_chart_single_model(
    model: str,
    haystack_size: int = None,
    needles: int = None,
    title: str = "",
    filename: str = "output.png",
):
    """
    Draws a chart comparing the performance of a single model for the retrieval task.
    Saves the chart with the specified filename and title.
    If haystack_size and needles are both None, it will draw charts for all matching JSON files.
    If only haystack_size is specified, it will draw one line per needle configuration.
    If only needles is specified, it will draw one line per haystack configuration.
    If needles is specified, the x-axis changes to absolute positions based on the largest haystack size.
    """
    # Find all matching JSON files for the specified model
    file_pattern = f"results/{model}-*-*.json"
    all_files = glob.glob(file_pattern)

    # Filter files based on haystack_size and needles if provided
    filtered_files = []
    for file in all_files:
        try:
            parts = os.path.basename(file).split("c-")
            file_haystack_size = int(parts[0].split("-")[-1])
            file_needles = int(parts[1].split("-")[0])

            if haystack_size is not None and file_haystack_size != haystack_size:
                continue
            if needles is not None and file_needles != needles:
                continue
            filtered_files.append((file, file_haystack_size))
        except (IndexError, ValueError):
            print(f"Skipping invalid file: {file}; parts: {parts}")
            continue

    if not filtered_files:
        print(f"No matching JSON files found for model {model}.")
        return

    # Determine the largest haystack size for scaling
    max_haystack_size = max(size for _, size in filtered_files)

    # Prepare a DataFrame to store all data for plotting
    plot_data = pd.DataFrame()

    for file, file_haystack_size in filtered_files:
        # Load JSON data
        try:
            with open(file, "r") as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"File not found: {file}")
            continue

        # Scale x-axis relative to the largest haystack size
        scaling_factor = file_haystack_size / max_haystack_size

        # Convert data into a DataFrame format suitable for plotting, with score as %
        file_needles = int(os.path.basename(file).split("c")[1].split("-")[1])
        model_df = pd.DataFrame(
            {
                "Relative Position in Phone Book (Scaled)": [
                    (int(k) / (len(data) - 1)) * scaling_factor for k in data.keys()
                ],
                "Absolute Position (Tokens)": [
                    int(k) * (file_haystack_size / (len(data) - 1)) for k in data.keys()
                ],
                "Score (%)": [v * 100 for v in data.values()],
                "Haystack Size": file_haystack_size,
                "Needles": file_needles,
                "Model": model,
            }
        )
        # Append to the main DataFrame
        plot_data = pd.concat([plot_data, model_df], ignore_index=True)

    # Set up the plot
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 8))

    # Set hue and x-axis dynamically dynamically based on input
    if haystack_size is not None:
        hue = "Needles"
        x_column = "Relative Position in Phone Book (Scaled)"
        x_label = "Relative Position in Phone Book (Scaled, 0 - 1)"
    elif needles is not None:
        hue = "Haystack Size"
        x_column = "Absolute Position (Tokens)"
        x_label = "Absolute position in input context (phone book size)"
    else:
        hue = "Haystack Size"
        x_column = "Relative Position in Phone Book (Scaled)"
        x_label = "Relative Position in Phone Book (Scaled, 0 - 1)"

    # Scatter plot with lower opacity
    sns.scatterplot(
        data=plot_data,
        x=x_column,
        y="Score (%)",
        hue=hue,
        alpha=0.3,
        legend="full",
    )

    # Plot polynomial regression lines for each needle configuration
    group_by = "Needles" if haystack_size is not None else "Haystack Size"
    for group_value in plot_data[group_by].unique():
        group_data = plot_data[plot_data[group_by] == group_value]
        x = group_data[x_column]
        y = group_data["Score (%)"]

        # Fit 2-degree polynomial
        poly_2 = np.polyfit(x, y, 2)
        y_pred_2 = np.polyval(poly_2, x)
        mse_2 = mean_squared_error(y, y_pred_2)

        # Fit 3-degree polynomial
        poly_3 = np.polyfit(x, y, 3)
        y_pred_3 = np.polyval(poly_3, x)
        mse_3 = mean_squared_error(y, y_pred_3)

        # Choose the better fit
        if mse_2 < mse_3:
            better_fit = 2
            best_poly = poly_2
        else:
            better_fit = 3
            best_poly = poly_3

        # Plot the better fit
        x_fit = np.linspace(x.min(), x.max(), 500)
        y_fit = np.polyval(best_poly, x_fit)

        plt.plot(
            x_fit,
            y_fit,
            label=f"{group_by}: {group_value} (best fit: {better_fit}-deg)",
            linewidth=2.5,
        )

    # Customize the plot
    plt.title(
        title if title else f"Performance of Model {model}",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    plt.xlabel(x_label)
    plt.ylabel("Score (%)")
    plt.xlim(0, max_haystack_size if needles is not None else 1)
    plt.ylim(0, 100)

    # Place legend below the plot, outside the plot area, arranged horizontally
    plt.legend(
        title="Model",
        loc="upper center",
        bbox_to_anchor=(
            0.5,
            -0.15,
        ),  # Position the legend outside the plot at the bottom
        ncol=3,  # Arrange legend items in a row (adjust ncol as needed)
        frameon=False,  # Optional: Remove the box around the legend
    )
    # Adjust layout to make room for the legend outside the plot
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save the plot
    output_dir = "agg_charts"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Chart saved as {output_path}")


if __name__ == "__main__":
    # models = [
    #     "gemini-1.5-flash-8b",
    #     "gemini-1.5-flash-002",
    #     "gemini-1.5-pro-002",
    #     "gpt-4o-2024-08-06",
    #     "gpt-4-turbo-2024-04-09",
    #     "gpt-4o-mini-2024-07-18",
    #     "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    #     "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    #     "meta-llama/Llama-3.2-3B-Instruct-Turbo",
    #     "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
    #     "Qwen/Qwen2.5-72B-Instruct-Turbo",
    #     "claude-3-5-haiku-20241022",
    #     "claude-3-5-sonnet-20241022",
    #     "claude-3-opus-20240229",
    # ]
    models = [
        # "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        # "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        # "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
        # "gemini-1.5-flash-8b",
        "gemini-1.5-flash-002",
        # "gemini-1.5-pro-002",
        "gemini-2.0-flash-exp",
        # "gpt-4o-2024-08-06",
        # "gpt-4-turbo-2024-04-09",
        # "gpt-4o-mini-2024-07-18",
        # "claude-3-5-haiku-20241022",
        # "claude-3-5-sonnet-20241022",
        # "claude-3-opus-20240229",
        # "claude-3-5-sonnet-20241022",
        # "ai21-jamba-1-5-large",
        # "ai21-jamba-1-5-mini",
    ]
    draw_chart(models, 5000, 100, file_prefix="gemini_flash_")

    # draw_chart_single_model(
    #     model="claude-3-5-haiku-20241022",
    #     # haystack_size=5000,
    #     needles=100,
    #     title="Performance for Claude 3 Haiku for different haystack sizes (100 needles)",
    #     filename="claude-3-haiku-100-needles.png",
    # )
