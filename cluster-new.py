import csv
import openai
import logging
import json
import re
import os
from typing import List, Dict, Tuple
from tqdm import tqdm
from collections import defaultdict

# Set up logging
logging.basicConfig(filename='kay_jewelers_categorization.log', level=logging.DEBUG, 
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
        parsed_json = json.loads(response_content)
        if isinstance(parsed_json, list):
            return parsed_json
        elif isinstance(parsed_json, dict):
            return [parsed_json]
        else:
            raise ValueError("Unexpected JSON structure")
    except json.JSONDecodeError:
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
                if 'ring' in ngram or 'engagement' in ngram:
                    ngram_categories['Rings'].append(ngram)
                elif 'necklace' in ngram or 'pendant' in ngram:
                    ngram_categories['Necklaces'].append(ngram)
                elif 'earring' in ngram:
                    ngram_categories['Earrings'].append(ngram)
                elif 'bracelet' in ngram:
                    ngram_categories['Bracelets'].append(ngram)
                elif 'watch' in ngram:
                    ngram_categories['Watches'].append(ngram)
                elif 'diamond' in ngram or 'gemstone' in ngram:
                    ngram_categories['Gemstones'].append(ngram)
                else:
                    ngram_categories['Other'].append(ngram)
    except Exception as e:
        logging.error(f"Error processing ngram file: {str(e)}")
        raise
    
    return ngram_counts, dict(ngram_categories)

def load_progress(file_path: str) -> Tuple[List[Dict], int]:
    if not os.path.exists(file_path):
        return [], 0
    
    with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        data = list(reader)
    
    completed = sum(1 for item in data if item.get('Main Category') and item.get('Main Category') != 'Uncategorized')
    logging.info(f"Loaded {completed} categorized keywords from {file_path}")
    return data, completed

def save_progress(keywords: List[Dict], output_file: str, completed: int):
    temp_file = f"temp_{output_file}"
    fieldnames = list(keywords[0].keys())
    
    with open(temp_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(keywords)
    
    os.replace(temp_file, output_file)
    logging.info(f"Saved progress ({completed} keywords) to {output_file}")

def categorize_keywords(keywords: List[Dict], main_categories: Dict[str, List[str]], ngram_counts: Dict[str, int], ngram_categories: Dict[str, List[str]], start_index: int = 0, batch_size: int = 10, save_interval: int = 100) -> List[Dict[str, str]]:
    categorized_keywords = []
    
    category_prompt = "\n".join([f"{cat}: {', '.join(subcats)}" for cat, subcats in main_categories.items()])
    ngram_info = "\n".join([f"{cat}: {', '.join(ngrams[:5])}" for cat, ngrams in ngram_categories.items()])
    
    for i in tqdm(range(start_index, len(keywords), batch_size), desc="Categorizing keywords"):
        batch = keywords[i:i+batch_size]
        prompt = f"""As an AI assistant for Kay Jewelers e-commerce website, categorize each keyword into a main category and subcategory based on the following structure:

{category_prompt}

Consider the following n-gram information for context:

{ngram_info}

For each keyword, return a JSON object with 'main_category' and 'subcategory'. If a keyword doesn't fit, use 'Other' for both. Consider the keyword's relevance to the jewelry industry, its current ranking on the site, and its relation to the provided n-grams.

Keywords (with intent, price info, and position):
{', '.join([f"{kw['Keyword']} (Intent: {kw['Keyword Intents']}, Price: {'$' + kw['Keyword'].split('$')[1] if '$' in kw['Keyword'] else 'N/A'}, Position: {kw['Position']}, N-gram count: {sum([ngram_counts.get(ngram, 0) for ngram in kw['Keyword'].lower().split() if ngram in ngram_counts])})" for kw in batch])}"""

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",  # Use the correct model name
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that categorizes keywords for Kay Jewelers e-commerce website, considering their current rankings and n-gram information."},
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
            
            for keyword, category in zip(batch, categories[:len(batch)]):
                keyword['Main Category'] = category['main_category']
                keyword['Subcategory'] = category['subcategory']
        except Exception as e:
            logging.error(f"Error processing batch: {str(e)}")
            categorized_keywords.extend([{'main_category': 'Uncategorized', 'subcategory': 'Uncategorized'}] * len(batch))
            for keyword in batch:
                keyword['Main Category'] = 'Uncategorized'
                keyword['Subcategory'] = 'Uncategorized'
        
        # Save progress
        if (i + batch_size) % save_interval == 0 or (i + batch_size) >= len(keywords):
            save_progress(keywords, 'categorized_kay_jewelers_keywords_in_progress.csv', i + batch_size)
    
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
    input_file = 'kay-20240918-semrush.csv'
    output_file = 'categorized_kay_jewelers_keywords.csv'
    progress_file = 'categorized_kay_jewelers_keywords_in_progress.csv'
    ngram_file = 'ngrams.csv'
    save_interval = 100  # Save every 100 keywords processed
    
    main_categories = {
        "Rings": ["Engagement Rings", "Wedding Bands", "Promise Rings", "Fashion Rings"],
        "Necklaces": ["Pendants", "Chains", "Lockets", "Chokers"],
        "Earrings": ["Studs", "Hoops", "Drops", "Climbers"],
        "Bracelets": ["Tennis Bracelets", "Charm Bracelets", "Bangles", "Cuffs"],
        "Watches": ["Men's Watches", "Women's Watches", "Smart Watches", "Luxury Watches"],
        "Gemstones": ["Diamonds", "Birthstones", "Precious Gems", "Semi-Precious Gems"],
        "Collections": ["Personalized Jewelry", "Signature Collections", "Gifts", "Sale Items"],
        "Services": ["Jewelry Repair", "Ring Sizing", "Engraving", "Cleaning"],
        "Other": ["Accessories", "Gift Cards", "Miscellaneous"]
    }
    
    logging.info("Starting keyword categorization process for Kay Jewelers")
    
    try:
        keywords, start_index = load_progress(progress_file)
        if not keywords:
            keywords = read_csv(input_file)
            if not keywords:
                logging.error("No data read from the CSV file. Please check the file contents.")
                return
        
        ngram_counts, ngram_categories = process_ngrams(ngram_file)
        categorize_keywords(keywords, main_categories, ngram_counts, ngram_categories, start_index=start_index, save_interval=save_interval)
        
        # Final save
        save_progress(keywords, output_file, len(keywords))
        
        logging.info(f"Categorized keywords saved to {output_file}")
        print(f"Categorized keywords saved to {output_file}")
    
    except Exception as e:
        logging.error(f"An error occurred during the categorization process: {str(e)}")
        print(f"An error occurred. Please check the log file for details.")

if __name__ == "__main__":
    main()