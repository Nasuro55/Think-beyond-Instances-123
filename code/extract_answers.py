import json
import os

def extract_answers_to_txt(jsonl_file, txt_file):
    if not os.path.exists(jsonl_file):
        print(f"❌ Error: File not found {jsonl_file}")
        return

    print(f"Processing {jsonl_file} ...")
    
    count = 0
    with open(jsonl_file, 'r', encoding='utf-8') as f_in, \
         open(txt_file, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            if not line.strip():
                continue
            
            try:
                data = json.loads(line)
                
                # 1. Prioritize 'final_answer' (field shown in reference images)
                # 2. If not found, check 'answer' (if it's a previously converted version)
                # 3. If neither exists, check 'solution' (as a fallback, though usually long)
                ans_content = None
                
                if 'final_answer' in data:
                    ans_content = data['final_answer']
                elif 'answer' in data:
                    ans_content = data['answer']
                elif 'solution' in data:
                    # Note: 'solution' is usually long; if only the final value is needed, this is usually not taken
                    ans_content = data['solution']
                
                # Handle data format
                # If final_answer is a list format like ["2"], we need to extract the string inside
                if isinstance(ans_content, list):
                    if len(ans_content) > 0:
                        ans_content = ans_content[0] # Extract the first element of the list
                    else:
                        ans_content = ""
                
                # Ensure conversion to string and write, removing leading/trailing whitespace
                final_str = str(ans_content).strip()
                f_out.write(final_str + "\n")
                count += 1
                
            except json.JSONDecodeError:
                print(f"⚠️ Skipping unparsable line")
                continue

    print(f"✅ Extraction complete!")
    print(f"📄 Extracted {count} answer lines in total")
    print(f"💾 Result saved to: {txt_file}")

if __name__ == "__main__":
    input_filename = "College_math.jsonl"
    output_filename = "College_math.txt"
    
    extract_answers_to_txt(input_filename, output_filename)