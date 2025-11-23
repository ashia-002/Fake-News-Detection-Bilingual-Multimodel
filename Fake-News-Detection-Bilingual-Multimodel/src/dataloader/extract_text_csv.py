import os
import csv
import re
from google.colab import userdata
from google.genai import Client
from getpass import getpass
from PIL import Image
import time
from tqdm.notebook import tqdm

# --- Configuration ---
# Main data directory path
MAIN_DIR = "/content/drive/MyDrive/fake-news-multimodal/data/"
# Output CSV file path
OUTPUT_CSV_PATH = "/content/drive/MyDrive/fake-news-multimodal/textdata/extracted_text_data.csv"

# Define the set of strings that indicate a failed or error extraction
ERROR_STRINGS = {"[QUOTA FAILED]", "[EXTRACTION FAILED]", "[FILE ERROR]", "[API KEY/PERMISSION ERROR]", "[MAX RETRIES EXCEEDED]"}


# --- 1. Initialize Gemini Client ---
try:
    # 1. Try to load the key directly from the Colab Secrets manager
    API_KEY = userdata.get('GEMINI_API_KEY')

    if API_KEY:
        print("Gemini API Key successfully loaded from Colab Secrets.")

    else:
        # Fallback: If not set in secrets, prompt the user (optional, but good practice)
        print("\nAPI Key not found in Colab Secrets. Please enter it manually.")
        API_KEY = getpass("Gemini API Key: ")

    # 2. Initialize the client using the retrieved key
    client = Client(api_key=API_KEY)
    print("Gemini Client initialized successfully.")

except Exception as e:
    print(f"Error initializing client: {e}")
    # If the client fails to initialize, the script cannot continue
    client = None

# --- 2. Extraction Function with Rate Limit Handling ---
def extract_text_from_image(image_path, client, max_retries=5):
    """
    Uses the Gemini API to extract text from a single image file,
    implementing intelligent backoff for rate limit errors (429).
    """
    prompt = "Extract ALL text (including Bengali and English) from this image exactly as it appears. Return only the extracted text."

    # Open the image file outside the retry loop to avoid repeated opening
    try:
        img = Image.open(image_path)
    except Exception as e:
        print(f"Error opening image file {os.path.basename(image_path)}: {e}")
        return "[FILE ERROR]"

    filename = os.path.basename(image_path)

    for attempt in range(max_retries):
        try:
            # Call the Gemini API
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[prompt, img]
            )
            return response.text.strip() # Success! Exit the retry loop.

        except Exception as e:
            error_message = str(e)

            # --- Rate Limit (429) Handling ---
            if '429 RESOURCE_EXHAUSTED' in error_message:

                # 1. Default long delay for safety
                retry_delay = 35.0
                # 2. Use regex to find the suggested retry delay in the error message
                match = re.search(r'Please retry in ([\d\.]+)s\.', error_message)

                if match:
                    try:
                        # Add a 1-second buffer to the suggested delay
                        retry_delay = float(match.group(1)) + 1.0
                    except:
                        pass # Use the default 35s if parsing fails

                if attempt == max_retries - 1:
                    print(f"Extraction failed for {filename} after {max_retries} attempts (Quota).")
                    return "[QUOTA FAILED]"

                print(f"Extraction for {filename} hit 429. Waiting {retry_delay:.2f}s (Attempt {attempt + 1}/{max_retries})...")
                time.sleep(retry_delay)

            # --- General Error Handling (API Key, Network, etc.) ---
            elif 'API_KEY_INVALID' in error_message or 'PERMISSION_DENIED' in error_message:
                # Non-recoverable errors: exit immediately
                print(f"Extraction failed for {filename} due to API Key/Permission Error.")
                return "[API KEY/PERMISSION ERROR]"

            else:
                # Handle other API/network errors with standard exponential backoff
                wait_time = (2 ** attempt) + 1 # 2^0+1=2s, 2^1+1=3s, 2^2+1=5s, ...
                if attempt == max_retries - 1:
                    print(f"Extraction failed for {filename} after {max_retries} attempts (General Error: {error_message.splitlines()[0]}).")
                    return "[EXTRACTION FAILED]"

                print(f"Extraction for {filename} failed (General Error). Retrying in {wait_time:.1f}s (Attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait_time)

    # If the loop finishes without a successful return
    return "[MAX RETRIES EXCEEDED]"


# --- 3. Main Extraction and CSV Generation Logic ---
def process_dataset_to_csv(main_dir, output_csv_path, client):
    """Loops through dataset, extracts text, and writes to CSV."""

    # 3.1 Gather all file paths and labels (Corrected for nested structure)
    all_data_points = []

    for label_dir in ['fake', 'real']:
        label_path = os.path.join(main_dir, label_dir)

        if not os.path.isdir(label_path):
            print(f"Warning: Main label directory not found: {label_path}")
            continue

        # Iterate through the language subfolders ('BN' and 'EN')
        for lang_dir in ['BN', 'EN']:
            lang_path = os.path.join(label_path, lang_dir)

            if not os.path.isdir(lang_path):
                print(f"Warning: Language directory not found: {lang_path}")
                continue

            # Get all image files in the deepest directory
            image_files = [
                os.path.join(lang_path, f)
                for f in os.listdir(lang_path)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]

            for file_path in image_files:
                # The classification label is the main folder name ('fake' or 'real')
                all_data_points.append({'path': file_path, 'label': label_dir})

    print(f"Found {len(all_data_points)} images to process across all subfolders.")

    # --- FIX: Ensure output directory exists before writing the file ---
    output_dir = os.path.dirname(output_csv_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Ensured output directory exists: {output_dir}")

    # 3.2 Robust Resume Logic: Read existing CSV, filter successful/failed, and prepare queue
    successful_rows = []
    failed_paths = []
    successful_paths_set = set()

    if os.path.exists(output_csv_path) and os.path.getsize(output_csv_path) > 0:
        try:
            with open(output_csv_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Check if the text is one of the error strings
                    if row.get('extracted_text') in ERROR_STRINGS or not row.get('extracted_text'):
                        failed_paths.append(row['image_path'])
                    else:
                        successful_rows.append(row)
                        successful_paths_set.add(row['image_path'])

            if successful_rows:
                print(f"Found {len(successful_rows)} previously successful entries.")
            if failed_paths:
                print(f"Found {len(failed_paths)} failed entries. They will be retried first.")

        except Exception as e:
            print(f"Error reading existing CSV for resume: {e}. Starting all extractions from scratch.")

    # 3.3 Create the final queue of images to process

    # 1. Start with the failed paths
    data_points_to_process = []

    # 2. Add failed paths back into the queue for a retry (must get the original label too)
    for dp in all_data_points:
        if dp['path'] in failed_paths:
            data_points_to_process.append(dp)

    # 3. Append all remaining, never-before-processed images
    for dp in all_data_points:
        if dp['path'] not in successful_paths_set and dp['path'] not in failed_paths:
            data_points_to_process.append(dp)

    # Calculate the starting index for the progress bar (based on successfully completed files)
    initial_count = len(successful_rows)
    total_count = len(all_data_points)

    if not data_points_to_process:
        print("\nAll files have been successfully processed!")
        return

    # 3.4 Overwrite the CSV with only the successful rows, then proceed to append
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['image_path', 'label', 'extracted_text']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(successful_rows) # Write the clean, successful data

    # Open CSV file again for appending the new results
    with open(output_csv_path, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['image_path', 'label', 'extracted_text']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # 3.5 Process and write data
        for data_point in tqdm(
            data_points_to_process,
            initial=initial_count,
            total=total_count,
            desc="Extracting Text"
        ):

            extracted_text = extract_text_from_image(data_point['path'], client)

            writer.writerow({
                'image_path': data_point['path'],
                'label': data_point['label'],
                'extracted_text': extracted_text
            })

            # --- Proactive Throttling ---
            # Introduce a small, fixed delay after every successful extraction
            time.sleep(1)

    print(f"\nData extraction complete. Results saved to: {output_csv_path}")

# --- 4. Execution ---
if __name__ == "__main__":
    if 're' not in globals():
        import re

    if 'client' in locals() and client is not None:
        process_dataset_to_csv(MAIN_DIR, OUTPUT_CSV_PATH, client)
    else:
        print("Skipping execution because the Gemini Client failed to initialize.")