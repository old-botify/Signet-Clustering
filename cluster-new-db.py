import csv
import openai
import logging
import json
import re
from typing import List, Dict, Tuple
from tqdm import tqdm
from collections import defaultdict

# Set up logging
logging.basicConfig(filename='keyword_categorization.log', level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Set up OpenAI API key
openai.api_key = ""

def read_csv(file_path: str) -> List[Dict]:
    logging.info(f"Attempting to read CSV file: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8-sig') as csvfile:
            reader = csv.DictReader(csvfile)
            data = list(reader)
            logging.info(f"Successfully read {len(data)} rows from the CSV file")
            logging.info(f"Columns in CSV: {', '.join(reader.fieldnames)}")
            return data
    except Exception as e:
        logging.error(f"Error reading CSV file: {str(e)}")
        raise

def parse_categories(response_content: str) -> List[Dict[str, str]]:
    try:
        # Try to parse the entire response as JSON
        parsed_json = json.loads(response_content)
        if isinstance(parsed_json, list):
            return parsed_json
        elif isinstance(parsed_json, dict):
            return [parsed_json]
        else:
            raise ValueError("Unexpected JSON structure")
    except json.JSONDecodeError:
        # If JSON parsing fails, try to extract JSON objects
        json_objects = re.findall(r'\{[^}]+\}', response_content)
        if json_objects:
            return [json.loads(obj) for obj in json_objects]
        else:
            logging.error(f"Failed to parse categories from response: {response_content}")
            return []

def process_ngrams(ngram_file: str) -> Tuple[Dict[str, int], Dict[str, List[str]]]:
    ngram_counts = {}
    ngram_categories = defaultdict(list)
    
    try:
        with open(ngram_file, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                ngram = row['ngram']
                count = int(row['count'])
                ngram_counts[ngram] = count
                
                # Categorize n-grams
                if 'wedding' in ngram or 'bridal' in ngram:
                    ngram_categories['Wedding Attire'].append(ngram)
                elif 'dress' in ngram or 'gown' in ngram:
                    ngram_categories['Dresses'].append(ngram)
                elif 'prom' in ngram:
                    ngram_categories['Prom'].append(ngram)
                elif 'bridesmaid' in ngram:
                    ngram_categories['Bridesmaid'].append(ngram)
                elif 'mother of' in ngram:
                    ngram_categories['Mother of the Bride/Groom'].append(ngram)
                elif 'accessories' in ngram or 'shoes' in ngram or 'veil' in ngram:
                    ngram_categories['Accessories'].append(ngram)
                else:
                    ngram_categories['Other'].append(ngram)
    except Exception as e:
        logging.error(f"Error processing ngram file: {str(e)}")
        raise
    
    return ngram_counts, dict(ngram_categories)

def categorize_keywords(keywords: List[Dict], main_categories: Dict[str, List[str]], ngram_counts: Dict[str, int], ngram_categories: Dict[str, List[str]], batch_size: int = 10) -> List[Dict[str, str]]:
    categorized_keywords = []
    
    category_prompt = "\n".join([f"{cat}: {', '.join(subcats)}" for cat, subcats in main_categories.items()])
    ngram_info = "\n".join([f"{cat}: {', '.join(ngrams[:5])}" for cat, ngrams in ngram_categories.items()])
    
    for i in tqdm(range(0, len(keywords), batch_size), desc="Categorizing keywords"):
        batch = keywords[i:i+batch_size]
        prompt = f"""As an AI assistant for an ecommerce bridal website, categorize each keyword into a main category and subcategory based on the following structure:

{category_prompt}

Consider the following n-gram information for context:

{ngram_info}

For each keyword, return a JSON object with 'main_category' and 'subcategory'. If a keyword doesn't fit, use 'Other' for both. Consider the keyword's relevance to the bridal industry, its current ranking on the site, and its relation to the provided n-grams.

Keywords (with intent, price info, and position):
{', '.join([f"{kw['Keyword']} (Intent: {kw['Keyword Intents']}, Price: {'$' + kw['Keyword'].split('$')[1] if '$' in kw['Keyword'] else 'N/A'}, Position: {kw['Position']}, N-gram count: {sum([ngram_counts.get(ngram, 0) for ngram in kw['Keyword'].lower().split() if ngram in ngram_counts])})" for kw in batch])}"""

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",  # Use the correct model name
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that categorizes keywords for an ecommerce bridal website, considering their current rankings and n-gram information."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            content = response.choices[0].message['content'].strip()
            logging.debug(f"API Response: {content}")
            
            categories = parse_categories(content)
            if len(categories) != len(batch):
                logging.warning(f"Mismatch in number of categories ({len(categories)}) and keywords ({len(batch)}). Using available categories.")
                categories.extend([{'main_category': 'Uncategorized', 'subcategory': 'Uncategorized'}] * (len(batch) - len(categories)))
            categorized_keywords.extend(categories[:len(batch)])
        except Exception as e:
            logging.error(f"Error processing batch: {str(e)}")
            categorized_keywords.extend([{'main_category': 'Uncategorized', 'subcategory': 'Uncategorized'}] * len(batch))
    
    return categorized_keywords

def write_csv(file_path: str, data: List[Dict], fieldnames: List[str]):
    logging.info(f"Attempting to write CSV file: {file_path}")
    try:
        with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        logging.info(f"Successfully wrote {len(data)} rows to the CSV file")
    except Exception as e:
        logging.error(f"Error writing CSV file: {str(e)}")
        raise

def main():
    input_file = '/Users/jameslange/Desktop/cluster/march2023-davids_bridal-20240917-semrush.csv'
    output_file = 'march2023_categorized_keywords.csv'
    ngram_file = 'ngrams.csv'  # Changed to match your file name
    
    main_categories = {
        "Bridal Attire": ["Wedding Dresses", "Budget Wedding Dresses", "Designer Wedding Dresses", "Plus Size Wedding Dresses"],
        "Bridesmaid Dresses": ["Bridesmaid Dresses", "Junior Bridesmaid Dresses"],
        "Special Occasion Dresses": ["Prom Dresses", "Homecoming Dresses", "Quinceanera Dresses", "Formal Event Dresses"],
        "Mother of the Bride/Groom": ["Mother of the Bride Dresses", "Mother of the Groom Dresses"],
        "Accessories": ["Veils", "Tiaras", "Wedding Shoes", "Bridal Jewelry", "Bridal Undergarments"],
        "Wedding Planning": ["Venues", "Catering", "Photography", "Decorations", "Invitations"],
        "Bridal Services": ["Dress Alterations", "Bridal Styling", "Beauty Services"],
        "Other": ["Miscellaneous", "Brand-specific"]
    }
    
    logging.info("Starting keyword categorization process for ecommerce bridal site")
    
    try:
        # Read input CSV
        keywords = read_csv(input_file)
        
        if not keywords:
            logging.error("No data read from the CSV file. Please check the file contents.")
            return
        
        # Process n-grams
        ngram_counts, ngram_categories = process_ngrams(ngram_file)
        
        # Categorize keywords
        categorized = categorize_keywords(keywords, main_categories, ngram_counts, ngram_categories)
        
        # Combine original data with categories
        for keyword, category in zip(keywords, categorized):
            keyword['Main Category'] = category['main_category']
            keyword['Subcategory'] = category['subcategory']
        
        # Write output CSV
        fieldnames = list(keywords[0].keys())
        write_csv(output_file, keywords, fieldnames)
        
        logging.info(f"Categorized keywords saved to {output_file}")
        print(f"Categorized keywords saved to {output_file}")
    
    except Exception as e:
        logging.error(f"An error occurred during the categorization process: {str(e)}")
        print(f"An error occurred. Please check the log file for details.")

if __name__ == "__main__":
    main()