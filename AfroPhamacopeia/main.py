# python
from ollama import chat
from utils import AfricanPharmacopoeiaRecipeList
from datasets import load_dataset
from huggingface_hub import HfApi, create_repo
import pandas as pd
import tqdm
import os

# Configure your Hugging Face dataset repo (change to your namespace)
REPO_ID = "jonathansuru/african_pharmacopoeia_recipes"
DATA_SUBDIR = "data"
FINAL_CSV = "african_pharmacopoeia_recipes.csv"

# Initialize and ensure the repo exists (requires `huggingface-cli login` beforehand)
api = HfApi()
create_repo(REPO_ID, repo_type="dataset", exist_ok=True)

# Load source dataset (requires access to the dataset on the Hub)
ds = load_dataset("jonathansuru/pharmacopeia_pdf")
texts = ds["train"]["text"]

# Prepare accumulators
records = []

# Prepare final CSV (stream-append)
if os.path.exists(FINAL_CSV):
    os.remove(FINAL_CSV)

for i, text in tqdm.tqdm(enumerate(texts), total=len(texts)):
    try:
        response = chat(
                messages=[{
                        "role": "user",
                        "content": text,
                }],
                model="qwen3:4b",
                format=AfricanPharmacopoeiaRecipeList.model_json_schema(),
        )

        result = AfricanPharmacopoeiaRecipeList.model_validate_json(response.message.content)
        records.extend(result.model_dump()["pharmacopoeia_recipe"])

        # Flush and upload every 1000 items (skip i == 0)
        if i > 0 and i % 500 == 0:
            data = pd.DataFrame.from_records(records)

            # Save and upload batch CSV
            batch_csv = f"african_pharmacopoeia_recipes_{i}.csv"
            data.to_csv(batch_csv, index=False)
            api.upload_file(
                    path_or_fileobj=batch_csv,
                    path_in_repo=f"{DATA_SUBDIR}/{batch_csv}",
                    repo_id=REPO_ID,
                    repo_type="dataset",
                    commit_message=f"Add batch {i}",
            )

            # Append to final CSV (header only once)
            write_header = not os.path.exists(FINAL_CSV)
            data.to_csv(FINAL_CSV, mode="a", header=write_header, index=False)

            # Reset batch records
            records = []
    except Exception as e:
        print(f"Error processing record {i}: {e}")
        continue

# Flush any remaining records and finalize
if records:
    data = pd.DataFrame.from_records(records)
    # Append remaining to final CSV
    write_header = not os.path.exists(FINAL_CSV)
    data.to_csv(FINAL_CSV, mode="a", header=write_header, index=False)

# Upload the final aggregated CSV
api.upload_file(
        path_or_fileobj=FINAL_CSV,
        path_in_repo=f"{DATA_SUBDIR}/{FINAL_CSV}",
        repo_id=REPO_ID,
        repo_type="dataset",
        commit_message="Add final dataset",
)
