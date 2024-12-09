from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
from datetime import datetime
import json
from pathlib import Path

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Required for flash messages

# Load the data
def load_data():
    try:
        # Load extraction results
        results_df = pd.read_csv('path_extraction_results_200.csv')
        
        # Load original reports to get the Notes
        original_reports_df = pd.read_csv(r"C:\Users\isult\Dropbox\AI\Data\pathology_1000_reports.csv")
        
        # Create a mapping dictionary for quick note lookup
        notes_dict = dict(zip(original_reports_df['Document_Number'], original_reports_df['Note']))
        
        # Add Notes to results DataFrame
        results_df['Note'] = results_df['Document_Number'].map(notes_dict)
        
        # Check if any evaluations exist
        if Path('human_evaluation_results.csv').exists():
            evaluated_df = pd.read_csv('human_evaluation_results.csv')
            # Get list of already evaluated document numbers
            evaluated_docs = evaluated_df['Document_Number'].tolist()
            # Filter out already evaluated reports
            results_df = results_df[~results_df['Document_Number'].isin(evaluated_docs)]
            
        # Drop rows where Note is missing
        results_df = results_df.dropna(subset=['Note'])
        
        if results_df.empty:
            print("No unevaluated reports found or all reports have been evaluated.")
            return pd.DataFrame()
            
        return results_df
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return pd.DataFrame()

# Save evaluation results
def save_evaluation(evaluation_data):
    try:
        # Convert evaluation data to DataFrame
        eval_df = pd.DataFrame([evaluation_data])
        
        # Add timestamp
        eval_df['evaluation_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Append to existing file or create new one
        if Path('human_evaluation_results.csv').exists():
            eval_df.to_csv('human_evaluation_results.csv', mode='a', header=False, index=False)
        else:
            eval_df.to_csv('human_evaluation_results.csv', index=False)
            
        return True
    except Exception as e:
        print(f"Error saving evaluation: {str(e)}")
        return False

@app.route('/', methods=['GET'])
def index():
    df = load_data()
    if df.empty:
        return render_template('no_reports.html')
    
    # Add progress tracking
    total_reports = len(df)
    current_index = total_reports - len(df) + 1
    percentage = round((current_index - 1) / total_reports * 100)
    
    # ... rest of your existing code ...
    
    return render_template('evaluate.html',
                          mrn=report_dict['MRN'],
                          doc_num=report_dict['Document_Number'],
                          entry_date=report_dict['Entry_Date'],
                          note=report_dict['Note'],
                          fields=fields_to_evaluate,
                          current_index=current_index,
                          total_reports=total_reports,
                          percentage=percentage)

@app.route('/submit_evaluation', methods=['POST'])
def submit_evaluation():
    try:
        # Get form data
        form_data = request.form
        
        # Create evaluation record
        evaluation = {
            'MRN': form_data['mrn'],
            'Document_Number': form_data['doc_num'],
            'Entry_Date': form_data['entry_date']
        }
        
        # Add evaluation results for each field
        for field in form_data:
            if field.startswith('field_'):
                field_name = field.replace('field_', '')
                evaluation[f'{field_name}_correct'] = field in request.form
                evaluation[f'{field_name}_value'] = form_data[f'value_{field_name}']
        
        # Save evaluation
        if save_evaluation(evaluation):
            flash('Evaluation saved successfully!', 'success')
        else:
            flash('Error saving evaluation!', 'error')
            
        return redirect(url_for('index'))
        
    except Exception as e:
        flash(f'Error: {str(e)}', 'error')
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, port=5000)