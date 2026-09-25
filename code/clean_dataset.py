import json
import re

def clean_answer(answer_str):
    """
    Cleaning function for mathematical dataset answers
    """
    if not isinstance(answer_str, str):
        answer_str = str(answer_str)
    
    # =======================================================
    # 1. Clean up garbage characters in LaTeX environment
    # =======================================================
    # Remove LaTeX negative spacing (\!) and small spacing (\,)
    answer_str = re.sub(r'\\[!,;]', '', answer_str)
    
    # Remove dollar signs \$ or $
    answer_str = answer_str.replace('\\$', '').replace('$', '')

    # =======================================================
    # 2. Fix units and text (\text, \mbox)
    # =======================================================
    
    # Common unit words that need to be completely removed
    units_to_remove = [
        'cm', 'm', 'inches', 'feet', 'degrees', 'cents', 'dollars', 
        'rupee', 'kuna', 'pula', 'sq\\.?\\s*units', 'units'
    ]
    
    def text_replacer(match):
        full_match = match.group(0)       # Get the full match "\text{...}"
        content = match.group(1).strip()  # Get content inside braces "..."
        
        # Check if content ends with a unit, or is a unit itself
        for unit in units_to_remove:
            # Match cases like " cm" or "5 cents" inside the text
            if re.search(r'\b' + unit + r'\b', content, re.IGNORECASE):
                # Try to keep the number part, remove the unit
                # Example: \text{5 cents} -> 5
                num_part = re.search(r'[\d\.]+', content)
                return num_part.group(0) if num_part else ''
        
        # [Modification Point]: If it is not a unit (e.g., person name "Navin"), 
        # return the complete original match string directly.
        # This preserves the \text{} wrapper.
        return full_match

    # Match \text{...} or \mbox{...}
    answer_str = re.sub(r'\\(?:text|mbox)\{([^\}]+)\}', text_replacer, answer_str)

    # Secondary cleanup: Sometimes units are not inside \text{}, but directly follow the number
    for unit in units_to_remove:
        answer_str = re.sub(r'\s+' + unit + r'\b', '', answer_str, flags=re.IGNORECASE)

    # =======================================================
    # 3. Fix degree symbols
    # =======================================================
    answer_str = re.sub(r'\^?\{?\\circ\}?', '', answer_str)

    # =======================================================
    # 4. Fix commas in numbers (thousands separator)
    # =======================================================
    # Core logic: Only remove commas that are "surrounded by digits" and 
    # "followed by exactly 3 digits on the right"
    while True:
        new_str = re.sub(r'(\d),\s*(\d{3})\b', r'\1\2', answer_str)
        if new_str == answer_str:
            break
        answer_str = new_str

    # =======================================================
    # 5. Fix spaces
    # =======================================================
    # replace(" ", "") will remove all spaces.
    # For preserved \text{ Navin }, this becomes \text{Navin}, complying with LaTeX standard format
    answer_str = answer_str.replace(' ', '')
    
    # =======================================================
    # 6. Final cleanup
    # =======================================================
    # Remove potential residual \boxed{} wrapper
    match_boxed = re.search(r'\\boxed\{(.*)\}', answer_str)
    if match_boxed:
        answer_str = match_boxed.group(1)
        
    return answer_str.strip()

# ==========================================
# Execute file processing
# ==========================================
input_file = './test.jsonl'        # Your original filename
output_file = './data.jsonl'       # Output filename

# Correction counter
fixed_count = 0

print("Starting data cleaning...")
with open(input_file, 'r', encoding='utf-8') as infile, \
     open(output_file, 'w', encoding='utf-8') as outfile:
    
    for line_num, line in enumerate(infile):
        if not line.strip():
            continue
        
        try:
            data = json.loads(line)
            if 'answer' in data:
                original = data['answer']
                cleaned = clean_answer(original)
                
                if original != cleaned:
                    fixed_count += 1
                    # Print fix details containing 'text' for inspection
                    if '\\text' in original:
                         print(f"Line {line_num+1} Fixed: [{original}]  ->  [{cleaned}]")
                
                data['answer'] = cleaned
            
            outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
            
        except json.JSONDecodeError:
            print(f"Error parsing line {line_num}")

print(f"\nCleaning completed! Fixed {fixed_count} items. Results saved to {output_file}")