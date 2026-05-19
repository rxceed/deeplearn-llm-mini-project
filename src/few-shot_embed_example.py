import ollama
import pandas as pd
import json
import math
from pathlib import Path

from sklearn.metrics import accuracy_score, cohen_kappa_score, mean_squared_error
try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:
    # Fallback for scikit-learn < 1.4
    root_mean_squared_error = None

def load_few_shot_examples(data_path: Path, num_examples: int = 12):
    """
    Creates balanced few-shot examples extracted from the training data, 
    matching the logic in create_modelfile.py.
    """
    df = pd.read_csv(data_path)
    
    # Calculate how many examples we need per score category to be balanced
    unique_scores = df['score'].unique()
    num_scores = len(unique_scores)
    target_per_score = max(1, num_examples // num_scores)
    
    # Sample balanced examples
    # We use pd.concat with a list comprehension to ensure the 'score' column is preserved
    sampled_df = pd.concat([
        group.sample(n=min(target_per_score, len(group)), random_state=42) 
        for _, group in df.groupby('score')
    ]).reset_index(drop=True)
    
    # Shuffle the final dataset
    sampled_df = sampled_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    few_shot_messages = []
    for _, ex in sampled_df.iterrows():
        # Clean newlines to keep the message compact
        cleaned_content = str(ex['full_text']).replace('\n', ' ').replace('\r', '')
        few_shot_messages.append({
            'role': 'user',
            'content': f"Essay ID: {ex['essay_id']} | Essay content: {cleaned_content}"
        })
        # Format the assistant response as JSON
        assistant_data = {"essay_id": ex['essay_id'], "score": int(ex['score'])}
        few_shot_messages.append({
            'role': 'assistant',
            'content': json.dumps(assistant_data)
        })
    return few_shot_messages

def read_csv(data_path: Path, sampling: bool=False, sampling_count: int=15):
    load = pd.read_csv(data_path)
    if sampling == True:
        df = load.sample(sampling_count)
    else:
        df = load
    essay_id = df.get('essay_id')
    essay_content = df.get('full_text')
    essay_score = df.get('score')
    return essay_id, essay_content, essay_score

def call_llm(essay_id, essay_content, few_shot_messages):
    system_prompt = """You are a judge at an essay writing competition. 
                        Assess based on Holistic essay scoring in scale of 1 to 6, with the worst being 1 and the best being 6.
                        Asses the mastery of the writer by judging the quality of the essay, reasoning, evidence, focus, coherence, vocabulary, 
                        and accuracy in grammar and sentence structure.
                        Give bonus the stronger those criterias are and give penalty the weaker those criterias are.
                        Give minor penalty for seldom typos, but massive penalty if the typos are too frequent. Give major penalty if the content can't be assembled into proper paragraph.
                        After assessing the mastery, use the available reference tool to determine the score of the essay.
                        Answer in JSON format with keys 'essay_id' and 'score'.
                        Example: {"essay_id": "8b2bead", "score": 3}
                        Answer only in one line."""
    
    messages = [{'role': 'system', 'content': system_prompt}]
    
    # Prepend few-shot examples
    messages.extend(few_shot_messages)
    
    # Add the target essay
    messages.append({
        'role': 'user',
        'content': f"Essay ID: {essay_id} | Essay content: {essay_content}"
    })

    # Call the base model since we are embedding the few-shots manually
    response = ollama.chat(
        model='gemma4', 
        messages=messages,
        options={"temperature": 0.2},
        format="json",
    )

    print(response['message']['content'])
    return response['message']['content']

def extract_score(llm_response: str) -> int:
    try:
        data = json.loads(llm_response)
        if 'score' in data:
            return int(data['score'])
        for val in data.values():
            if str(val).isdigit():
                return int(val)
    except (json.JSONDecodeError, ValueError, TypeError):
        try:
            parts = llm_response.split(',')
            if len(parts) >= 2:
                return int(parts[1].strip())
        except Exception:
            pass
    print(f"Warning: Failed to parse score from response: {llm_response}")
    return -1

def process_all_essays(data_path: Path, training_path: Path, output_path: Path, num_few_shot: int = 12):
    print(f"Loading {num_few_shot} balanced few-shot examples...")
    few_shot_messages = load_few_shot_examples(training_path, num_few_shot)
    
    essay_ids, essay_contents, scores = read_csv(data_path, True, 20)
    results = []
    
    y_true = []
    y_pred = []
    
    for eid, content, real_score in zip(essay_ids, essay_contents, scores):
        print(f"Processing Essay ID: {eid}")
        llm_response = call_llm(eid, content, few_shot_messages)
        
        predicted_score = extract_score(llm_response)
        actual_score = int(real_score)
        
        print(f"predicted score: {predicted_score}")
        print(f"real score: {actual_score}\n---")
        
        if predicted_score != -1:
            y_pred.append(predicted_score)
            y_true.append(actual_score)
        
        results.append({
            'essay_id': eid,
            'real_score': actual_score,
            'llm_response': llm_response,
            'predicted_score': predicted_score
        })
        
        if len(results) == 100:
            break
        
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
    
    if y_true and y_pred:
        accuracy = accuracy_score(y_true, y_pred)
        qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
        rmse = math.sqrt(mean_squared_error(y_true, y_pred)) if root_mean_squared_error is None else root_mean_squared_error(y_true, y_pred)
            
        print(f"Total Processed Successfully: {len(y_true)}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Quadratic Weighted Kappa (QWK): {qwk:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    else:
        print("No valid scores to calculate metrics.")

if __name__ == '__main__':
    output_dir = Path('./dataset/result')
    output_dir.mkdir(exist_ok=True, parents=True)
    data_path = Path('./dataset/split/test.csv')
    training_path = Path('./dataset/split/train.csv')
    output_path = Path('./dataset/result/results_fewshot_test.csv')
    process_all_essays(data_path, training_path, output_path, num_few_shot=60)