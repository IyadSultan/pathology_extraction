from flask import Flask, render_template, request, redirect, url_for, flash, session
import pandas as pd
from datetime import datetime
import json
from pathlib import Path

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

def load_data(current_index=0):
    try:
        # Load extraction results
        results_df = pd.read_csv('path_extraction_results_200.csv')
        
        # Load original reports to get the Notes
        original_reports_df = pd.read_csv(r"E:\Dropbox\AI\Data\pathology_1000_reports.csv")
        
        # Create a mapping dictionary for quick note lookup
        notes_dict = dict(zip(original_reports_df['Document_Number'], original_reports_df['Note']))
        
        # Add Notes to results DataFrame
        results_df['Note'] = results_df['Document_Number'].map(notes_dict)
        
        # Drop rows where Note is missing
        results_df = results_df.dropna(subset=['Note'])
        
        # Load previous evaluations if they exist
        submitted_status = {}
        previous_evaluations = {}
        if Path('human_evaluation_results.csv').exists():
            evaluated_df = pd.read_csv('human_evaluation_results.csv')
            for _, row in evaluated_df.iterrows():
                doc_num = str(row['Document_Number'])
                submitted_status[doc_num] = True
                # Store field evaluations
                previous_evaluations[doc_num] = {col: row[col] 
                                               for col in row.index 
                                               if not col in ['MRN', 'Document_Number', 'Entry_Date', 'evaluator_name', 'evaluation_timestamp']}
            
        if results_df.empty:
            print("No reports found.")
            return pd.DataFrame(), submitted_status, previous_evaluations
            
        return results_df, submitted_status, previous_evaluations
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return pd.DataFrame(), {}, {}

def save_evaluation(evaluation_data):
    try:
        # Convert evaluation data to DataFrame
        eval_df = pd.DataFrame([evaluation_data])
        
        # Add timestamp
        eval_df['evaluation_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Append to existing file or create new one
        if Path('human_evaluation_results.csv').exists():
            # Read existing evaluations
            existing_df = pd.read_csv('human_evaluation_results.csv')
            # Update or append based on Document_Number
            doc_num = evaluation_data['Document_Number']
            if doc_num in existing_df['Document_Number'].values:
                existing_df.loc[existing_df['Document_Number'] == doc_num] = eval_df.iloc[0]
                existing_df.to_csv('human_evaluation_results.csv', index=False)
            else:
                eval_df.to_csv('human_evaluation_results.csv', mode='a', header=False, index=False)
        else:
            eval_df.to_csv('human_evaluation_results.csv', index=False)
            
        return True
    except Exception as e:
        print(f"Error saving evaluation: {str(e)}")
        return False

@app.route('/', methods=['GET'])
def index():
    return redirect(url_for('evaluate', index=0))

@app.route('/evaluate/<int:index>', methods=['GET'])
def evaluate(index):
    if 'evaluator_name' not in session:
        return redirect(url_for('set_evaluator'))
    
    # Load the data
    df, submitted_status, previous_evaluations = load_data()
    
    if df.empty:
        return render_template('no_reports.html')
    
    total_reports = len(df)
    index = max(0, min(index, total_reports - 1))
    
    current_report = df.iloc[index]
    doc_num = str(current_report['Document_Number'])
    
    # Extract fields to evaluate
    fields_to_evaluate = {col: current_report[col] for col in current_report.index 
                         if col not in ['MRN', 'Document_Number', 'Entry_Date', 'Note']}
    
    # Get previous evaluation results if they exist
    field_results = {}
    if doc_num in previous_evaluations:
        prev_eval = previous_evaluations[doc_num]
        field_results = {field: prev_eval.get(field, True) for field in fields_to_evaluate.keys()}
    
    # Handle potential missing or NaN values
    mrn = str(current_report['MRN']) if 'MRN' in current_report else 'N/A'
    entry_date = str(current_report['Entry_Date']) if 'Entry_Date' in current_report else 'N/A'
    note = str(current_report['Note']) if 'Note' in current_report else 'Note not found'
    
    return render_template('evaluate.html',
                          mrn=mrn,
                          doc_num=doc_num,
                          entry_date=entry_date,
                          note=note,
                          fields=fields_to_evaluate,
                          field_results=field_results,
                          current_index=index + 1,
                          total_reports=total_reports,
                          percentage=round((index + 1) / total_reports * 100),
                          evaluator_name=session['evaluator_name'],
                          is_submitted=doc_num in submitted_status,
                          has_previous=index > 0,
                          has_next=index < total_reports - 1,
                          prev_index=index - 1,
                          next_index=index + 1)

@app.route('/submit_evaluation', methods=['POST'])
def submit_evaluation():
    try:
        form_data = request.form
        
        # Create evaluation record with metadata
        evaluation = {
            'MRN': form_data['mrn'],
            'Document_Number': form_data['doc_num'],
            'Entry_Date': form_data['entry_date'],
            'evaluator_name': session['evaluator_name']
        }
        
        # Get all field names from hidden values and add True/False results
        field_names = [key.replace('value_', '') for key in form_data.keys() 
                      if key.startswith('value_')]
        
        for field_name in field_names:
            checkbox_name = f'field_{field_name}'
            evaluation[field_name] = checkbox_name in form_data
        
        if save_evaluation(evaluation):
            flash('Evaluation saved successfully!', 'success')
        else:
            flash('Error saving evaluation!', 'error')
        
        # Move to next report after submission
        current_index = int(request.form.get('current_index', 0))
        return redirect(url_for('evaluate', index=current_index))
        
    except Exception as e:
        flash(f'Error: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/set_evaluator', methods=['GET', 'POST'])
def set_evaluator():
    if request.method == 'POST':
        evaluator_name = request.form.get('evaluator_name')
        if evaluator_name:
            session['evaluator_name'] = evaluator_name
            return redirect(url_for('index'))
    return render_template('set_evaluator.html')

@app.route('/skip_report', methods=['GET'])
def skip_report():
    flash('Report skipped!', 'info')
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, port=5000)