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

def call_llm(essay_id, essay_content):
    messages = [{
        'role': 'system',
        'content': """You are a judge at an essay writing competition. 
                        Assess based on Holistic essay scoring in scale of 1 to 6, with the worst being 1 and the best being 6.
                        Asses the mastery of the writer by judging the quality of the essay, reasoning, evidence, focus, coherence, vocabulary, 
                        and accuracy in grammar and sentence structure.
                        Give bonus the stronger those criterias are and give penalty the weaker those criterias are.
                        Give minor penalty for seldom typos, but massive penalty if the typos are too frequent. Give major penalty if the content can't be assembled into proper paragraph.
                        Answer in JSON format with keys 'essay_id' containing essay_id and 'score' containing the numeric value.
                        Example: {\"essay_id\": 8b2bead,\"score\": 3}
                        essay_id is the same as the input essay_id and score is the numeric value of the score you give to the essay.
                        Answer only in one line"""
    }, {
        'role': 'user',
        'content': f"Essay ID: {essay_id}\nEssay content: {essay_content}"
    }]

    # First call to let the model decide to use the tool
    response = ollama.chat(
        model='dlm-aes-few-shot',
        messages=messages,
        options={"temperature": 0.2},
        format="json",
    )

    print(response['message']['content'])
    return response['message']['content']

def extract_score(llm_response: str) -> int:
    try:
        # The LLM is configured to output JSON.
        data = json.loads(llm_response)
        if 'score' in data:
            return int(data['score'])
        
        # Fallback if the json structure is slightly different but contains values
        for val in data.values():
            if str(val).isdigit():
                return int(val)
                
    except (json.JSONDecodeError, ValueError, TypeError):
        # Fallback to the old comma-split method just in case JSON format failed completely
        try:
            parts = llm_response.split(',')
            if len(parts) >= 2:
                return int(parts[1].strip())
        except Exception:
            pass
            
    print(f"Warning: Failed to parse score from response: {llm_response}")
    return -1  # Indicate parsing failure

def process_all_essays(data_path: Path, output_path: Path):
    essay_ids, essay_contents, scores = read_csv(data_path)
    results = []
    
    y_true = []
    y_pred = []
    
    # Using zip to iterate through the arrays returned by read_csv
    for eid, content, real_score in zip(essay_ids, essay_contents, scores):
        print(f"Processing Essay ID: {eid}")
        llm_response = call_llm(eid, content)
        
        predicted_score = extract_score(llm_response)
        actual_score = int(real_score)
        
        print(f"predicted score: {predicted_score}")
        print(f"real score: {actual_score}\n---")
        
        # Only add to metrics if parsing was successful
        if predicted_score != -1:
            y_pred.append(predicted_score)
            y_true.append(actual_score)
        
        results.append({
            'essay_id': eid,
            'real_score': actual_score,
            'llm_response': llm_response,
            'predicted_score': predicted_score
        })
        
        if len(results) == 100: # Keeping the existing break condition
            break
        
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
    
    if y_true and y_pred:
        accuracy = accuracy_score(y_true, y_pred)
        qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
        
        if root_mean_squared_error is not None:
            rmse = root_mean_squared_error(y_true, y_pred)
        else:
            rmse = math.sqrt(mean_squared_error(y_true, y_pred))
            
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
    output_path = Path('./dataset/result/results_fewshot_test.csv')
    process_all_essays(data_path, output_path)