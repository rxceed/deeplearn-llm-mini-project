import pandas as pd
from pathlib import Path
import json

def create_ollama_modelfile(data_path: Path, output_path: Path, base_model: str = "gemma2:9b", num_examples: int = 1200):
    """
    Creates an Ollama Modelfile with balanced few-shot examples extracted from the training data.
    """
    print(f"Reading training data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Calculate how many examples we need per score category to be balanced
    unique_scores = df['score'].unique()
    num_scores = len(unique_scores)
    target_per_score = num_examples // num_scores
    
    print(f"Targeting approximately {target_per_score} examples per score category (Total: {num_examples})...")
    
    # Sample balanced examples
    # We use pd.concat with a list comprehension to ensure the 'score' column is preserved across different pandas versions
    sampled_df = pd.concat([group.sample(n=min(target_per_score, len(group)), random_state=42) for _, group in df.groupby('score')]).reset_index(drop=True)
    
    # Shuffle the final dataset so the model doesn't just see them in order of score
    sampled_df = sampled_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Actually sampled {len(sampled_df)} examples.")
    print("Distribution:")
    print(sampled_df['score'].value_counts().sort_index())
    
    # System prompt (aligned with main.py and flattened to a single line)
    system_prompt = (
        "You are a judge at an essay writing competition."
        "Assess based on Holistic essay scoring in scale of 1 to 6, with the worst being 1 and the best being 6."
        "Asses the mastery of the writer by judging the quality of the essay, reasoning, evidence, focus, coherence, vocabulary," 
        "and accuracy in grammar and sentence structure."
        "Give bonus the stronger those criterias are and give penalty the weaker those criterias are."
        "Give minor penalty for seldom typos, but massive penalty if the typos are too frequent. Give major penalty if the content can't be assembled into proper paragraph."
        "Answer in JSON format with keys 'essay_id' containing essay_id and 'score' containing the numeric value."
        "Example: {\"essay_id\": 8b2bead,\"score\": 3}"
        "essay_id is the same as the input essay_id and score is the numeric value of the score you give to the essay."
        "Answer only in one line"
    )

    modelfile_content = [
        f"FROM {base_model}",
        f'SYSTEM "{system_prompt}"',
        # Set some parameters if needed
        "PARAMETER temperature 0.3",
        "PARAMETER stop \"\\n---\""
    ]

    # Add few-shot messages
    for _, ex in sampled_df.iterrows():
        # Clean newlines from essay content to keep it on one line in the Modelfile
        cleaned_content = str(ex['full_text']).replace('\n', ' ').replace('\r', '').replace('"', '\\"')
        user_msg = f"Essay ID: {ex['essay_id']} | Essay content: {cleaned_content}"
        
        # Format the assistant response as JSON
        assistant_data = {"essay_id": ex['essay_id'], "score": int(ex['score'])}
        assistant_msg = json.dumps(assistant_data).replace('"', '\\"')
        
        modelfile_content.append(f'MESSAGE user "{user_msg}"')
        modelfile_content.append(f'MESSAGE assistant "{assistant_msg}"')

    # Write to file
    with open(output_path, 'w') as f:
        f.write("\n".join(modelfile_content))
    
    print(f"\nModelfile created successfully at {output_path}")
    print("\nTo create the model in Ollama, run:")
    print(f"ollama create gemma4-fewshot -f {output_path}")

if __name__ == "__main__":
    DATA_PATH = Path("./dataset/split/train.csv")
    OUTPUT_PATH = Path("gemma4-fewshot.Modelfile")
    
    if not DATA_PATH.exists():
        print(f"Error: Dataset not found at {DATA_PATH}")
    else:
        create_ollama_modelfile(DATA_PATH, OUTPUT_PATH, base_model="gemma4", num_examples=60)
