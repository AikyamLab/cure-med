import pandas as pd
import os
from tabulate import tabulate  # For pretty table output (install if needed: pip install tabulate)

# Define the output directory where your logic evaluated CSVs are saved
output_dir = os.path.join(os.getcwd(), "BaseModel_logic_evaluation")

# List of languages
languages = [
    "Amharic", "Bengali", "French", "Hausa", "Hindi", "Japanese", "Korean",
    "Spanish", "Swahili", "Thai", "Turkish", "Vietnamese", "Yoruba"
]

# Function to compute stats for logic
def compute_logic_stats():
    per_lang_results = []
    global_true = 0
    global_false = 0
    global_total = 0
    
    for lang in languages:
        csv_path = os.path.join(output_dir, f"{lang}_qwen2.5_14B_logic_evaluation.csv")
        
        if not os.path.exists(csv_path):
            print(f"Skipping {lang}: CSV not found.")
            continue
        
        df = pd.read_csv(csv_path)
        # Clean the Logic_Score column: convert to string, strip whitespace, normalize case
        df['Logic_Score'] = df['Logic_Score'].astype(str).str.strip().str.title()
        
        # Filter to only evaluated rows (non-null/empty after cleaning)
        df = df[df['Logic_Score'].notnull() & (df['Logic_Score'] != '')]
        
        total = len(df)
        if total == 0:
            print(f"Skipping {lang}: No evaluated samples.")
            continue
        
        # Optional debug: Print unique values in Logic_Score for this language
        # print(f"Unique Logic_Score values for {lang}: {df['Logic_Score'].unique()}")
        
        true_count = (df['Logic_Score'] == 'True').sum()
        false_count = (df['Logic_Score'] == 'False').sum()
        accuracy = (true_count / total) * 100 if total > 0 else 0.0
        
        per_lang_results.append({
            'Language': lang,
            'Total Samples': total,
            'Correct (True)': true_count,
            'Incorrect (False)': false_count,
            'Accuracy (%)': f"{accuracy:.2f}"
        })
        
        global_true += true_count
        global_false += false_count
        global_total += total
    
    global_accuracy = (global_true / global_total) * 100 if global_total > 0 else 0.0
    global_stats = {
        'Global Total Samples': global_total,
        'Global Correct (True)': global_true,
        'Global Incorrect (False)': global_false,
        'Global Accuracy (%)': f"{global_accuracy:.2f}"
    }
    
    return per_lang_results, global_stats

# Compute for Logic_Score
logic_per_lang, logic_global = compute_logic_stats()

# Display results
print("\nLogic Statistics Per Language:")
if logic_per_lang:
    print(tabulate(logic_per_lang, headers='keys', tablefmt='grid'))
else:
    print("No logic data available.")

print("\nLogic Global Statistics:")
print(tabulate([logic_global], headers='keys', tablefmt='grid'))