import pandas as pd
import json
from pathlib import Path

annotations = Path('social-processing/nt-rev2-all_pmids/report/corrected_all')

merged_annotations = {}

# Merge the three JSON files and extract the corresponding IDs from the CSV into single json
for category in ['false_negatives_abstract', 'false_negatives_fulltext', 'false_positives_fulltext']:
    with open(annotations / f'{category}.json') as f:
        data = json.load(f)
    
    for entry in data:
        pmid = entry['pmid']
        judgment = entry['judgment']
        
        if pmid not in merged_annotations:
            merged_annotations[pmid] = {
                'judgment': judgment,
                'comment': entry.get('comment', ''),
                'pmid': pmid,
                'category': category,
            }

    
# Judgment can be 'agree', 'disagree'
# if False Negative and 'disagree' -> YES otherwise NO
# if False Positive and 'disagree' -> NO otherwise YES

# Turn into two columns: 'include_in_posthoc' and 'rationale'
posthoc_results = {}

for pmid, info in merged_annotations.items():
    if 'false_negatives' in info['category']:
        include_in_posthoc = 'YES' if info['judgment'] == 'disagree' else 'NO'
    elif 'false_positives' in info['category']:
        include_in_posthoc = 'NO' if info['judgment'] == 'disagree' else 'YES'

    rationale = info['comment']
    posthoc_results[pmid] = {
        'posthoc_status': include_in_posthoc,
        'posthoc_reason': rationale
    }

df = pd.read_csv('../neurometabench/data/all_studies_annotated.csv')

# Load study_pmid as string to avoid issues with leading zeros
df['study_pmid'] = df['study_pmid'].astype(str)

# Map the posthoc results to the dataframe
df['posthoc_status'] = df['study_pmid'].map(lambda x: posthoc_results.get(x, {}).get('posthoc_status', ''))
df['posthoc_reason'] = df['study_pmid'].map(lambda x: posthoc_results.get(x, {}).get('posthoc_reason', ''))

df.to_csv('../neurometabench/data/all_studies_annotated.csv', index=False)

# Combine df['corrected_status'] and df['posthoc_status'], prefer 'posthoc_status' if not empty
def determine_final_status(row):
    if row['posthoc_status'] in ['YES', 'NO']:
        return row['posthoc_status']
    return row['corrected_status']

# Save the study_pmid values where final_status is 'YES' to a text file
df['final_status'] = df.apply(determine_final_status, axis=1)
yes_pmids = df[df['final_status'] == 'YES']['study_pmid'].tolist()

with open('social-processing/posthoc.txt', 'w') as f:
    f.write('\n'.join(yes_pmids))

with open('social-processing/all_pmids.txt', 'w') as f:
    f.write('\n'.join(df['study_pmid'].tolist()))