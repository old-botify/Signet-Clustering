import csv
import openai
import logging
import json
import re
from typing import List, Dict
from tqdm import tqdm

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
            reader.fieldnames = [field.strip().replace('\ufeff', '') for field in reader.fieldnames]
            data = list(reader)
            logging.info(f"Successfully read {len(data)} rows from the CSV file")
            logging.info(f"Columns in CSV: {', '.join(reader.fieldnames)}")
            return data
    except Exception as e:
        logging.error(f"Error reading CSV file: {str(e)}")
        raise

def parse_categories(response_content: str) -> List[Dict[str, str]]:
    try:
        # First, try to parse as JSON
        json_start = response_content.find('[')
        json_end = response_content.rfind(']') + 1
        if json_start != -1 and json_end != -1:
            json_content = response_content[json_start:json_end]
            return json.loads(json_content)
        else:
            raise json.JSONDecodeError("No JSON array found", response_content, 0)
    except json.JSONDecodeError:
        # If JSON parsing fails, try to extract categories manually
        categories = []
        pattern = r"'main_category':\s*'([^']*)',\s*'subcategory':\s*'([^']*)'"
        matches = re.findall(pattern, response_content)
        for match in matches:
            categories.append({
                'main_category': match[0],
                'subcategory': match[1]
            })
        if not categories:
            logging.error(f"Failed to parse categories from response: {response_content}")
        return categories

def categorize_keywords(keywords: List[Dict], main_categories: Dict[str, List[str]], batch_size: int = 10) -> List[Dict[str, str]]:
    categorized_keywords = []
    
    category_prompt = "\n".join([f"{cat}: {', '.join(subcats)}" for cat, subcats in main_categories.items()])
    
    for i in tqdm(range(0, len(keywords), batch_size), desc="Categorizing keywords"):
        batch = keywords[i:i+batch_size]
        prompt = f"""As an AI assistant for an ecommerce bridal website, categorize each keyword into a main category and subcategory based on the following structure:

{category_prompt}

For each keyword, return a JSON object with 'main_category' and 'subcategory'. If a keyword doesn't fit, use 'Other' for both. Consider the keyword's relevance to the bridal industry and its current ranking on the site.

Keywords (with intent, price info, and position):
{', '.join([f"{kw['Keyword']} (Intent: {kw['Keyword Intents']}, Price: {'$' + kw['Keyword'].split('$')[1] if '$' in kw['Keyword'] else 'N/A'}, Position: {kw['Position']})" for kw in batch])}"""

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",  # Use the correct model name
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that categorizes keywords for an ecommerce bridal website, considering their current rankings."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            content = response.choices[0].message['content'].strip()
            logging.debug(f"API Response: {content}")
            
            categories = parse_categories(content)
            if len(categories) != len(batch):
                raise ValueError(f"Mismatch in number of categories ({len(categories)}) and keywords ({len(batch)})")
            categorized_keywords.extend(categories)
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
    input_file = '/Users/jameslange/Desktop/cluster/davids_bridal-20240917-semrush.csv'
    output_file = 'improved_categorized_keywords.csv'
    
    main_categories = {
        "Bridal Attire": ["Wedding Dresses", "Budget Wedding Dresses", "Designer Wedding Dresses", "Plus Size Wedding Dresses", "Bridesmaid Dresses", "Flower Girl Dresses", "Mother of the Bride Dresses"],
        "Bridal Accessories": ["Veils", "Tiaras", "Wedding Shoes", "Bridal Jewelry", "Bridal Undergarments"],
        "Special Event Attire": ["Prom Dresses", "Quinceanera Dresses", "Formal Event Dresses"],
        "Wedding Planning": ["Venues", "Catering", "Photography", "Decorations", "Invitations"],
        "Bridal Services": ["Dress Alterations", "Bridal Styling", "Beauty Services"],
        "Price Categories": ["Under $100", "$100-$500", "$500-$1000", "Over $1000"],
        "SEO Performance": ["Top Ranking", "Mid Ranking", "Low Ranking", "Unranked"],
        "Other": ["Miscellaneous", "Brand-specific"]
    }
    
    logging.info("Starting keyword categorization process for ecommerce bridal site")
    
    try:
        # Read input CSV
        keywords = read_csv(input_file)
        
        if not keywords:
            logging.error("No data read from the CSV file. Please check the file contents.")
            return
        
        # Categorize keywords
        categorized = categorize_keywords(keywords, main_categories)
        
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