import csv
import json
import os

# Define file paths (adjust if needed)
questions_txt_path = "data/test/questions.txt"  # your filtered_questions.txt
csv_file_path = "data/test/TestSet.csv"
output_json_path = "data/test/reference_answers.json"

 # Load the list of questions from questions.txt
with open(questions_txt_path, "r", encoding="utf-8") as f:
    questions = [line.strip() for line in f if line.strip()]

# Load the CSV data
extended_data = []
with open(csv_file_path, "r", encoding="utf-8") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        extended_data.append(row)

# Create a mapping from question text to its extended metadata.
# Note: The CSV header for the question text is "question", not "Question"
extended_map = { row["question"].strip(): row for row in extended_data }

# Build the reference answers dictionary only for questions in your txt file.
# The keys are question IDs (as strings) and the values are dictionaries.
reference = {}
for idx, q in enumerate(questions, start=1):
    # Try to find a matching row in the extended data
    if q in extended_map:
        row = extended_map[q]
        reference[str(idx)] = {
            "Answer": row["answer"].strip(),
            "Answer Type": row["answer_type"].strip(),
            "Answer Restriction": row["answer_restriction"].strip(),
            "Event Fact": row["event_fact"].strip(),
            "CMU Pittsburgh": row["cmu_pittsburgh"].strip()
        }
    else:
        print(f"Warning: No matching CSV entry found for question: '{q}'")

# Save the resulting reference answers JSON
os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
with open(output_json_path, "w", encoding="utf-8") as out_file:
    json.dump(reference, out_file, indent=2)

print(f"Reference answers saved to {output_json_path}")