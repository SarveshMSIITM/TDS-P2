# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "argparse",
#     "fastapi",
#     "logging",
#     "matplotlib",
#     "openai",
#     "pandas",
#     "scikit-learn",
#     "scipy",
#     "seaborn",
#     "uvicorn",
#     "requests",
# ]
# ///
import argparse
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn
import io
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.cluster import KMeans, AgglomerativeClustering
import json
import os
import logging
from scipy.stats import zscore
import requests

# Set up logging
logging.basicConfig(level=logging.INFO)

# Define a function to parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Process CSV file.")
    parser.add_argument("csv_file", type=str, help="Path to the CSV file")
    return parser.parse_args()

# Define a function to read a CSV file
def read_csv(filepath):
    try:
        # Try reading the file with utf-8 encoding
        return pd.read_csv(filepath, encoding="utf-8")
    except UnicodeDecodeError:
        # If there's a UnicodeDecodeError, try reading the file with a different encoding
        return pd.read_csv(filepath, encoding="latin1")

# Define a function to generate a summary of a DataFrame
def generate_summary(df: pd.DataFrame) -> dict:
    buffer = io.StringIO()
    df.info(buf=buffer)
    summary = {
        "info": buffer.getvalue(),
        "description": df.describe().to_string(),
        "missing_values": (df.isnull().sum().to_string()),
    }
    return summary

# Parse command-line arguments and read the CSV file
args = parse_args()
df = read_csv(args.csv_file)

# Initialize OpenAI API (requires a valid API token in environment variables)
api_token = os.getenv("AIPROXY_TOKEN")
if not api_token:
    logging.error("OpenAI API token is not configured. Set 'AIPROXY_TOKEN' in your environment.")
    exit(1)

def call_openai_api(input_messages, model="gpt-4o-mini"):
    """
    Call the OpenAI API to process data and log the raw response for debugging.
    """
    url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    #url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": input_messages,
        "functions": [
            {
                "name": "suggest_columns",
                "description": "Suggest relevant columns for clustering or correlation analysis.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "columns": {
                            "type": "array",
                            "items": {"type": "string"},
                        }
                    },
                    "required": ["columns"],
                },
            }
        ],
        "function_call": {"name": "suggest_columns"},
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Decode the response JSON
        result = response.json()
        
        # Log the raw response for debugging
        #logging.info("OpenAI API Raw Response: %s", json.dumps(result, indent=4))

        # Process and validate the API response
        return process_openai_response(result)
    except requests.exceptions.RequestException as req_error:
        logging.error(f"Request error: {req_error}")
        raise
    except json.JSONDecodeError as json_error:
        logging.error(f"JSON decode error: {json_error}")
        raise
    except Exception as e:
        logging.error(f"Unhandled exception: {e}")
        raise

def process_openai_response(result):
    """
    Process the JSON-decoded response from the OpenAI API.
    """
    try:
        # Ensure 'choices' exists and is not empty
        if "choices" in result and len(result["choices"]) > 0:
            choice = result["choices"][0]
            
            # Check for 'message' in the first choice
            if "message" in choice and choice["message"]:
                message = choice["message"]
                
                # Handle 'function_call' if present
                if "function_call" in message:
                    function_call = message["function_call"]
                    arguments_str = function_call.get("arguments", None)
                    
                    if arguments_str:
                        try:
                            # Decode JSON arguments
                            arguments = json.loads(arguments_str)
                            return arguments  # Return successfully parsed arguments
                        except json.JSONDecodeError as e:
                            logging.error(f"Error decoding JSON: {e}")
                            return {"error": "Invalid JSON in function_call arguments"}
                    
                    return {"error": "No arguments found in function_call"}
                
                # Handle 'content' if no 'function_call'
                elif "content" in message and message["content"] is not None:
                    return {"content": message["content"]}
                else:
                    return {"error": "No usable content or function call in response"}
            else:
                return {"error": "No message found in choices"}
        else:
            return {"error": "OpenAI response does not contain 'choices' field or it is empty"}
    
    except KeyError as e:
        logging.error(f"KeyError: {e}")
        return {"error": f"Missing key: {e}"}
    except Exception as e:
        logging.error(f"Unhandled exception in processing: {e}")
        return {"error": str(e)}

def analyze_with_openai(summary: dict, analysis_type: str):
    """
    Send the data summary to OpenAI for analysis and get a structured response.
    """
    prompt = f"Given the following data summary: {summary}, suggest columns for {analysis_type} in JSON format."
    input_messages = [
        {"role": "system", "content": "You are an AI that performs data analysis."},
        {"role": "user", "content": prompt},
    ]

    try:
        response = call_openai_api(input_messages)
        # Parse the JSON response
        return response  # Return structured JSON data
    except json.JSONDecodeError as jde:
        logging.error(f"Error decoding JSON from OpenAI response: {jde}")
        return {"error": "Invalid JSON response"}
    except Exception as e:
        logging.error(f"Error in OpenAI analysis ({analysis_type}): {e}")
        return {"error": str(e)}
    
def generate_bubble_map(df: pd.DataFrame, clustering_response: dict):
    """
    Generate and save a bubble map based on clustering columns.

    Args:
        df (pd.DataFrame): The input DataFrame containing data for clustering.
        clustering_response (dict): A dictionary containing clustering column information.

    Returns:
        str: The file path of the saved bubble map image, or None if an error occurs.
    """
    try:
        # Ensure 'columns' exists in clustering_response and is valid
        clustering_columns = clustering_response.get("columns")
        
        if not clustering_columns or not isinstance(clustering_columns, list) or len(clustering_columns) < 2:
            logging.warning("Invalid or missing clustering columns in response.")
            return None

        # Check if the selected columns are valid and present in the DataFrame
        if all(col in df.columns for col in clustering_columns):
            # Handle missing values by imputing or dropping
            df_selected = df[clustering_columns]

            if df_selected.isnull().any().any():  # Check for NaNs
                logging.warning("NaN values detected. Imputing missing values with column means.")
                df_selected = df_selected.dropna()

            # Normalize the data using z-scores
            df_normalized = df_selected.apply(zscore)

            # Perform clustering (e.g., KMeans with 3 clusters for demonstration purposes)
            kmeans = KMeans(n_clusters=3, random_state=42)
            df_normalized['Cluster'] = kmeans.fit_predict(df_normalized)

            # Create a bubble map using the first two clustering columns and cluster labels
            plt.figure(figsize=(6, 6))
            sns.scatterplot(
                data=df_normalized,
                x=clustering_columns[0],
                y=clustering_columns[1],
                hue='Cluster',
                size=clustering_columns[2] if len(clustering_columns) > 2 else None,
                sizes=(50, 300),
                palette='viridis',
                alpha=0.7
            )
            plt.title("Bubble Map for Clustering")
            plt.xlabel(clustering_columns[0])
            plt.ylabel(clustering_columns[1])

            # Save the plot locally as a PNG file
            file_path = "clustering_bubble_map.png"
            plt.savefig(file_path, format='png')
            plt.clf()  # Clear the current figure
            plt.close()

            logging.info("Bubble map generated and saved successfully.")
            return file_path
        else:
            logging.warning("Suggested clustering columns are invalid or missing in the DataFrame.")
            return None
    except Exception as e:
        logging.error(f"Error generating bubble map: {e}")
        return None

def generate_barplot(df: pd.DataFrame, barplot_response: dict):
    """
    Generate and save a barplot based on the specified columns.

    Args:
        df (pd.DataFrame): The input DataFrame containing data for analysis.
        barplot_response (dict): A dictionary containing column information for the barplot.

    Returns:
        str: The file path of the saved barplot image.
    """
    try:
        # Ensure 'columns' exists in barplot_response and is valid
        barplot_columns = barplot_response.get("columns")
        
        if not barplot_columns or not isinstance(barplot_columns, list) or len(barplot_columns) < 2:
            logging.warning("Invalid or missing bar plot columns in response.")
            return None

        # Check if the selected columns are valid and present in the DataFrame
        if all(col in df.columns for col in barplot_columns):
            # Select the first column for the x-axis and the second column for the y-axis
            x_col = barplot_columns[0]
            y_col = barplot_columns[1] if len(barplot_columns) > 1 else None

            if y_col is None:
                logging.warning("Barplot requires at least two columns: one for x and one for y.")
                return None

            # Handle missing values by dropping them
            df_selected = df[[x_col, y_col]].dropna()

            # Create a barplot using the selected columns
            plt.figure(figsize=(8, 6))
            sns.barplot(data=df_selected, x=x_col, y=y_col, palette="viridis")

            plt.title("Barplot Analysis")
            plt.xlabel(x_col)
            plt.ylabel(y_col)

            # Save the plot locally as a PNG file
            file_path = "barplot_analysis.png"
            plt.savefig(file_path, format='png')
            plt.clf()
            plt.close()

            logging.info("Barplot generated and saved successfully.")
            return file_path
        else:
            logging.warning("Suggested barplot columns are invalid or missing in the DataFrame.")
            return None
    except Exception as e:
        logging.error(f"Error generating barplot: {e}")
        return None

def generate_correlation_heatmap(df: pd.DataFrame, correlation_response: dict):
    """
    Generate and save a correlation heatmap based on specified columns.

    Args:
        df (pd.DataFrame): The input DataFrame containing data for analysis.
        correlation_response (dict): A dictionary containing column information for the correlation heatmap.

    Returns:
        str: The file path of the saved heatmap image.
    """
    try:
        # Ensure 'columns' exists in correlation_response and is valid
        columns = correlation_response.get("columns")
        
        if not columns or not isinstance(columns, list):
            logging.warning("Invalid or missing correlation columns in response.")
            return None

        # Check if the selected columns are valid and present in the DataFrame
        if all(col in df.columns for col in columns):
            # Select the relevant columns from the DataFrame
            df_selected = df[columns]

            # Calculate the correlation matrix
            corr_matrix = df_selected.corr()

            plt.figure(figsize=(8, 8))
            sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
            plt.title("Correlation Heatmap")
            plt.tight_layout()

            # Save the plot locally as a PNG file
            file_path = "correlation_heatmap.png"
            plt.savefig(file_path, format='png')
            plt.clf()
            plt.close()

            logging.info("Heatmap generated and saved successfully.")
            return file_path
        else:
            logging.warning("Suggested correlation columns are invalid or missing in the DataFrame.")
            return None
    except Exception as e:
        logging.error(f"Error generating correlation heatmap: {e}")
        return None

def generate_line_chart(df: pd.DataFrame, time_series_response: dict):
    """
    Generate and save a line chart based on the specified columns for time series analysis.

    Args:
        df (pd.DataFrame): The input DataFrame containing data for analysis.
        time_series_response (dict): A dictionary containing column information for the time series plot.

    Returns:
        str: The file path of the saved line chart image.
    """
    try:
        # Ensure 'columns' exists in time_series_response and is valid
        columns = time_series_response.get("columns")
        
        if not columns or not isinstance(columns, list):
            logging.warning("Invalid or missing time series columns in response.")
            return None

        # Check if the selected columns are valid and present in the DataFrame
        if all(col in df.columns for col in columns):
            # Select the relevant columns from the DataFrame
            df_selected = df[columns]

            # Create a line plot for each column
            plt.figure(figsize=(10, 6))
            for column in columns:
                plt.plot(df_selected.index, df_selected[column], label=column)

            plt.title("Time Series Line Chart")
            plt.xlabel("Date")
            plt.ylabel("Value")
            plt.legend()
            plt.tight_layout()

            # Save the plot locally as a PNG file
            file_path = "time_series_line_chart.png"
            plt.savefig(file_path, format='png')
            plt.clf()
            plt.close()

            logging.info("Line chart generated and saved successfully.")
            return file_path
        else:
            logging.warning("Suggested time series columns are invalid or missing in the DataFrame.")
            return None
    except Exception as e:
        logging.error(f"Error generating line chart: {e}")
        return None

def call_openai_api_for_story(summary, plot_file_paths, analysis_type="trend_analysis", model="gpt-4o-mini"):
    """
    Call the OpenAI API to generate a compelling story with a detailed analysis
    of the provided summary and plot images, with dynamic customization.
    """
    url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    #url = "https://api.openai.com/v1/chat/completions"

    # api_token = {api_token}  # Replace with your actual API token
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json",
    }

    # System message with dynamic task assignment
    system_message = {
        "role": "system",
        "content": (
            "You are a financial analyst and storyteller. Your task is to generate a clear, engaging narrative story"
            "based on a summary of financial data and accompanying visualizations. Provide insights, analyze trends, "
            "and highlight implications using the provided data. Adapt your analysis style based on the type of data. "
            "Additionally, make direct observations on the clarity, design, and effectiveness of the visualizations."
        ),
    }

    # Dynamic handling of the summary prompt
    user_summary_message = {
        "role": "user",
        "content": (
            f"Summary: {summary}\n\n"
            "Create an interesting and engaging story based on the data and observations"
            "Analyze the trends, patterns, and key insights in this summary. "
            "Provide specific recommendations based on the data provided. "
            "If there are noticeable patterns or anomalies, be sure to highlight them."
        ),
    }

    # Dynamic plot context handling: adapt to the number and type of plots
    plot_context = "\n".join([f"Plot {i+1}: {path}" for i, path in enumerate(plot_file_paths)])
    user_plots_message = {
        "role": "user",
        "content": (
            "Associated plots:\n" 
            f"{plot_context}\n\n"
            "Please incorporate relevant observations from these plots in your analysis. "
            "If the plot shows a trend, anomaly, or pattern, explain its relevance to the summary. "
            "For specific plot types, adjust the focus: for correlation heatmaps, analyze relationships; for time-series plots, "
            "highlight changes over time. Additionally, make direct observations on the clarity and design of each plot, "
            "commenting on how effectively the plots convey the insights and trends in the data."
        ),
    }

    # Recursive agentic feedback loop: initial analysis request
    def generate_analysis_request(iteration=1):
        if iteration == 1:
            return {
                "role": "user",
                "content": (
                    "Based on the summary and plots, craft a cohesive narrative focusing on trend analysis. "
                    "Your story should include the following:\n"
                    "1. High-level overview of the observed trends.\n"
                    "2. Key data points that showcase the trends.\n"
                    "3. Implications of these trends for forecasting or decision-making."
                ),
            }
        elif iteration == 2:
            return {
                "role": "user",
                "content": (
                    "Please revisit the previous analysis and focus more on any identified anomalies or unexpected patterns "
                    "that might have significant implications for decision-making."
                ),
            }
        else:
            return {
                "role": "user",
                "content": (
                    "Please refine your analysis by incorporating any feedback or additional insights based on the most recent observations. "
                    "Focus specifically on the relationship between identified trends and anomalies, and suggest actionable strategies."
                ),
            }

    # Initialize first iteration's request message
    iteration = 1
    analysis_request_message = generate_analysis_request(iteration)

    # Payload to send to the API for the first analysis
    payload = {
        "model": model,
        "messages": [
            system_message, 
            user_summary_message, 
            user_plots_message, 
            analysis_request_message
        ],
    }

    try:
        # Send the first request to the OpenAI API
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Decode the response JSON
        result = response.json()

        # Log the raw response for debugging
        logging.info("OpenAI API Raw Response: %s", json.dumps(result, indent=4))

        # Extract and return the story content for the first iteration
        story_content = result["choices"][0]["message"]["content"]
        logging.info(f"Story Content (Iteration {iteration}): {story_content}")

        # Optionally, refine the analysis in the next iteration
        iteration += 1
        analysis_request_message = generate_analysis_request(iteration)

        # Prepare the next request with refined analysis
        refined_payload = {
            "model": model,
            "messages": [
                system_message, 
                user_summary_message, 
                user_plots_message, 
                analysis_request_message
            ],
        }

        # Send the next request to refine the analysis (recursive iteration)
        refined_response = requests.post(url, headers=headers, json=refined_payload)
        refined_response.raise_for_status()  # Handle HTTP errors

        refined_result = refined_response.json()
        refined_story_content = refined_result["choices"][0]["message"]["content"]
        logging.info(f"Refined Story Content (Iteration {iteration}): {refined_story_content}")

        return refined_story_content

    except requests.exceptions.RequestException as req_error:
        logging.error(f"Request error: {req_error}")
        raise
    except json.JSONDecodeError as json_error:
        logging.error(f"JSON decode error: {json_error}")
        raise
    except KeyError as key_error:
        logging.error(f"Unexpected API response structure: {key_error}")
        raise
    except Exception as e:
        logging.error(f"Unhandled exception: {e}")
        raise

def write_story_to_readme(story, plot_file_paths):
    """
    Write the generated story along with the plot images to a README.md file.
    """
    with open("README.md", "w") as readme_file:
        readme_file.write("# Analysis Report\n\n")
        readme_file.write(story)
        readme_file.write("\n\n## Plot Images\n\n")

        for path in plot_file_paths:
            readme_file.write(f"![Plot Image]({path})\n\n")

    logging.info("Story and images written to README.md successfully.")

# Initialize FastAPI app
app = FastAPI()
@app.get("/", response_class=HTMLResponse)
def display_summary():
    try:
        #logging.info("Generating summary...")
        summary = generate_summary(df)
        #logging.info("Summary generated.")

        correlation_response = analyze_with_openai(summary, "correlation analysis")
        clustering_response = analyze_with_openai(summary, "clustering analysis")
        barplot_response = analyze_with_openai(summary, "barplot analysis")
        time_series_response = analyze_with_openai(summary, "time series analysis")

        # Display raw responses
        logging.info("Correlation Analysis Response:")
        logging.info(correlation_response)
        logging.info("Clustering Analysis Response:")
        logging.info(clustering_response)
        logging.info("Barplot Analysis Response:")
        logging.info(barplot_response)
        logging.info("Time Series Analysis Response:")
        logging.info(time_series_response)

        correlation_map = generate_correlation_heatmap(df, correlation_response)
        bubble_map = generate_bubble_map(df, clustering_response)
        barplot = generate_barplot(df, barplot_response)
        line_chart = generate_line_chart(df, time_series_response)

        # Collect plot file paths, only if they are not None
        plot_file_paths = [
            path for path in [correlation_map, bubble_map, barplot, line_chart] if path is not None
            ]


        try:
            # Call the OpenAI API to generate the story
            story = call_openai_api_for_story(summary, plot_file_paths)

            # Write the story and plot images to README.md
            write_story_to_readme(story, plot_file_paths)
        except Exception as e:
            logging.error(f"Failed to generate story and write to README.md: {e}")


        return f"""
        <h1>Data Analysis Summary</h1>
        <h2>Summary Info:</h2>
        <pre>{summary["info"]}</pre>
        <h2>Missing Values:</h2>
        <pre>{summary["missing_values"]}</pre>
        <h2>Correlation Columns:</h2>
        <pre>{correlation_response["columns"]}</pre>
        <h2>Clustering Suggested Columns:</h2>
        <pre>{clustering_response["columns"]}</pre>
        <h2>Correlation Heatmap:</h2>
        <img src="correlation_heatmap.png" alt="Correlation Heatmap" />
        """
    except Exception as e:
        logging.error(f"Error in summary generation: {e}")
        return f"<h1>Error:</h1><p>{e}</p>"

if __name__ == "__main__":
    display_summary()

    