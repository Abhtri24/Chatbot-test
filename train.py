from openai import OpenAI
from dotenv import load_dotenv
import os
import time
import sys

print("Starting script...")
print(f"Current working directory: {os.getcwd()}")
print(f"Files in directory: {os.listdir('.')}")

# Load environment variables from key.env
print("\nLoading environment variables...")
load_dotenv('key.env')
api_key = os.getenv("OPENAI_API_KEY")

print(f"API Key found: {'Yes' if api_key else 'No'}")
if api_key:
    print(f"API Key length: {len(api_key)}")
    print(f"API Key starts with: {api_key[:10]}...")

if not api_key:
    print("Error: OPENAI_API_KEY not found in key.env file")
    sys.exit(1)

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

def prepare_training_file():
    try:
        # Check if file exists
        if not os.path.exists('electrostore_finetune_dataset.jsonl'):
            print("Error: electrostore_finetune_dataset.jsonl file not found")
            sys.exit(1)
            
        print("Opening training file...")
        with open('electrostore_finetune_dataset.jsonl', 'rb') as file:
            print("Uploading file to OpenAI...")
            response = client.files.create(
                file=file,
                purpose='fine-tune'
            )
            print(f"File upload response: {response}")
            return response.id
    except Exception as e:
        print(f"Error uploading file: {str(e)}")
        sys.exit(1)

def create_fine_tune(file_id):
    try:
        print("Creating fine-tuning job...")
        response = client.fine_tuning.jobs.create(
            training_file=file_id,
            model="gpt-3.5-turbo"
        )
        print(f"Fine-tuning job response: {response}")
        return response.id
    except Exception as e:
        print(f"Error creating fine-tuning job: {str(e)}")
        sys.exit(1)

def check_fine_tune_status(job_id):
    try:
        print("\nFine-tuning progress:")
        print("-" * 50)
        while True:
            job = client.fine_tuning.jobs.retrieve(job_id)
            print(f"\nCurrent Status: {job.status}")
            
            if job.status == 'validating_files':
                print("✓ Validating your training data...")
            elif job.status == 'queued':
                print("✓ Files validated, waiting in queue...")
            elif job.status == 'running':
                print("✓ Training in progress...")
                if hasattr(job, 'trained_tokens'):
                    print(f"Tokens trained: {job.trained_tokens}")
            
            if job.status == 'succeeded':
                print("\n🎉 Fine-tuning completed successfully!")
                print(f"Your fine-tuned model name is: {job.fine_tuned_model}")
                return job.fine_tuned_model
            elif job.status == 'failed':
                print(f"\n❌ Fine-tuning failed! Error: {job.error if hasattr(job, 'error') else 'Unknown error'}")
                return None
                
            print("-" * 50)
            time.sleep(60)  # Check every minute
    except Exception as e:
        print(f"Error checking fine-tuning status: {str(e)}")
        return None

def main():
    print("\nStarting fine-tuning process...")
    print(f"Using API key: {api_key[:8]}...{api_key[-4:]}")
    
    # Step 1: Upload training file
    print("\nUploading training file...")
    file_id = prepare_training_file()
    print(f"File uploaded successfully. File ID: {file_id}")
    
    # Step 2: Create fine-tuning job
    print("\nCreating fine-tuning job...")
    job_id = create_fine_tune(file_id)
    print(f"Fine-tuning job created. Job ID: {job_id}")
    
    # Step 3: Monitor fine-tuning progress
    print("\nMonitoring fine-tuning progress...")
    model_name = check_fine_tune_status(job_id)
    
    if model_name:
        print("\nFine-tuning completed successfully!")
        print(f"Your fine-tuned model name is: {model_name}")
        print("\nUpdate the MODEL variable in tuning.py with this model name.")
    else:
        print("\nFine-tuning failed. Please check the error messages above.")

if __name__ == "__main__":
    main() 