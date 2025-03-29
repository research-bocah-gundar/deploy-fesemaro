import requests
import os
import sys
from tqdm import tqdm

# --- Configuration ---
# !!! IMPORTANT: Replace this with the ACTUAL URL to your model weights !!!
# Example (e.g., from Hugging Face Hub - ensure it's a direct download link):
# MODEL_WEIGHTS_URL = "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors?download=true"
MODEL_WEIGHTS_URL = "https://files.riqgarden.pp.ua/api/public/dl/aACU3P1n/Downloads/best_model_blg.safetensors"

MODEL_FILENAME = "best_model_blg.safetensors"
DOWNLOAD_DIR = os.getcwd() # Download to the current working directory
# Or specify a different directory:
# DOWNLOAD_DIR = "/path/to/your/model/directory"

# --- Download Logic ---

def download_file(url, destination_path, filename):
    """Downloads a file from a URL to a destination path with progress."""
    temp_destination_path = destination_path + ".part" # Temporary file path

    print(f"Attempting to download '{filename}' from {url}...")
    try:
        with requests.get(url, stream=True, timeout=60) as response: # Added timeout
            response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

            total_size_in_bytes = int(response.headers.get('content-length', 0))
            block_size = 1024 * 1024 # 1 MB chunks

            progress_bar = tqdm(
                total=total_size_in_bytes,
                unit='iB',
                unit_scale=True,
                desc=f"Downloading {filename}",
                leave=True # Keep the bar after completion
            )

            with open(temp_destination_path, 'wb') as file:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    file.write(data)

            progress_bar.close()

            if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                print(f"\nERROR: Download incomplete. Expected {total_size_in_bytes} bytes, received {progress_bar.n} bytes.")
                if os.path.exists(temp_destination_path):
                    os.remove(temp_destination_path) # Clean up partial file
                return False
            else:
                # Download seems complete, rename temporary file to final name
                os.rename(temp_destination_path, destination_path)
                print(f"\nSuccessfully downloaded and saved '{filename}' to '{destination_path}'.")
                return True

    except requests.exceptions.Timeout:
        print(f"\nError: The request timed out while downloading {filename}.")
        if os.path.exists(temp_destination_path): os.remove(temp_destination_path)
        return False
    except requests.exceptions.RequestException as e:
        print(f"\nError downloading '{filename}': {e}")
        if os.path.exists(temp_destination_path): os.remove(temp_destination_path)
        return False
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        if os.path.exists(temp_destination_path): os.remove(temp_destination_path)
        return False
    finally:
        # Ensure progress bar is closed in case of early exit or error
        if 'progress_bar' in locals() and not progress_bar.disable:
            try:
                progress_bar.close()
            except Exception:
                pass # Ignore errors closing already closed/failed bar

def ensure_model_downloaded(model_url, filename, download_dir):
    """Checks if model exists, downloads if not."""
    # Ensure the target directory exists
    os.makedirs(download_dir, exist_ok=True)

    model_path = os.path.join(download_dir, filename)

    if os.path.exists(model_path):
        print(f"Model file '{filename}' already exists at '{model_path}'. Skipping download.")
        return model_path
    else:
        print(f"Model file '{filename}' not found locally.")
        if model_url == "YOUR_ACTUAL_MODEL_DOWNLOAD_URL_HERE" or not model_url:
             print("ERROR: MODEL_WEIGHTS_URL is not set correctly. Please update the script.")
             return None

        if download_file(model_url, model_path, filename):
            return model_path
        else:
            print(f"Failed to download '{filename}'.")
            return None

# --- Main Execution ---
if __name__ == "__main__":
    print(f"Checking for model: {MODEL_FILENAME}")
    print(f"Target directory: {DOWNLOAD_DIR}")

    final_model_path = ensure_model_downloaded(MODEL_WEIGHTS_URL, MODEL_FILENAME, DOWNLOAD_DIR)

    if final_model_path:
        print(f"\nModel is ready at: {final_model_path}")
        # --- You can now load your model using final_model_path ---
        # Example:
        # model = load_my_model(final_model_path)
        # print("Model loaded successfully (placeholder).")
        # ---
    else:
        print("\nModel download/check failed. Cannot proceed with inference.")
        sys.exit(1) # Exit with an error code if the model isn't available

    # Optional: Run again to show the skip logic works
    print("\n--- Running check again to test skip logic ---")
    ensure_model_downloaded(MODEL_WEIGHTS_URL, MODEL_FILENAME, DOWNLOAD_DIR)
    print("--- Check complete ---")