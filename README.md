# Keyword Clustering Framework

This repository contains a collection of tools and approaches for clustering keywords, particularly useful for SEO and content organization. The framework offers multiple clustering methods to suit different needs and data sizes.

## Repository Structure

- `cluster-hdbscan.py`: HDBSCAN-based clustering for large-scale keyword sets
- `cluster-new.py`: OpenAI GPT-based clustering for Kay Jewelers
- `cluster-new-db.py`: OpenAI GPT-based clustering for David's Bridal
- `keyword_clustering.py`: Basic OpenAI GPT-based clustering template

## Clustering Approaches

### 1. HDBSCAN-Based Clustering (cluster-hdbscan.py)

Best for: Large-scale keyword clustering with semantic understanding

Features:
- Uses SentenceTransformers for semantic embeddings
- Implements HDBSCAN clustering algorithm
- Supports both CPU and CUDA processing
- Generates interactive visualizations (treemap/sunburst)
- Handles Excel pivot table output
- Includes stemming options

Usage:
```bash
python cluster-hdbscan.py mycsv.csv --column_name "Keyword" --chart_type "treemap"
```

### 2. GPT-Based Clustering

Available in three variants:

#### a. Basic Template (keyword_clustering.py)
- Generic implementation for any keyword set
- Uses OpenAI's GPT models for categorization
- Batch processing support
- Basic error handling and logging

#### b. Kay Jewelers Specific (cluster-new.py)
- Customized for jewelry industry taxonomy
- Includes n-gram processing
- Progress tracking and save points
- Detailed category hierarchy

#### c. David's Bridal Specific (cluster-new-db.py)
- Tailored for bridal industry keywords
- Custom category structure
- Supports price and intent analysis

## Key Features Across All Methods

1. **Data Loading**
   - CSV file support with encoding detection
   - Handles various column name formats
   - Duplicate removal options

2. **Processing Options**
   - Batch processing for large datasets
   - Progress tracking and logging
   - Error handling and recovery

3. **Output Formats**
   - CSV output with categorization
   - Excel pivot tables (Windows only)
   - Visualization options (HDBSCAN version)

## Choosing the Right Approach

1. Use **HDBSCAN** when:
   - Processing large keyword sets (1000+ keywords)
   - Need automated, semantic-based clustering
   - Want visualization options
   - Don't require specific industry taxonomy

2. Use **GPT-based** clustering when:
   - Need industry-specific categorization
   - Want human-readable category names
   - Have specific taxonomies to follow
   - Working with smaller datasets
   - Need to consider commercial intent

## Installation

1. For HDBSCAN approach:
```bash
pip install chardet numpy pandas plotly typer pywin32 hdbscan sentence-transformers
```

2. For GPT-based approaches:
```bash
pip install openai tqdm logging pandas
```

## Configuration

### HDBSCAN Configuration:
```python
--min_cluster_size: int = 2
--model_name: str = "all-MiniLM-L6-v2"
--device: str = "cpu"
```

### GPT Configuration:
- Set your OpenAI API key
- Configure category hierarchies
- Adjust batch sizes for processing

## Performance Considerations

1. **HDBSCAN Approach**
   - RAM usage scales with dataset size
   - GPU acceleration available
   - Better for automated clustering

2. **GPT Approach**
   - API rate limits apply
   - Costs associated with API usage
   - Better for precise categorization

## Best Practices

1. **Data Preparation**
   - Clean your keyword data
   - Remove duplicates if needed
   - Ensure consistent formatting

2. **Processing**
   - Use appropriate batch sizes
   - Monitor memory usage
   - Save progress regularly

3. **Output Handling**
   - Validate clustering results
   - Review category assignments
   - Back up important results

## Logging

All methods include comprehensive logging:
- Error tracking
- Progress monitoring
- Performance metrics
- Results validation

## Contributing

To extend or modify:
1. Choose the appropriate base version
2. Maintain consistent error handling
3. Update documentation
4. Add tests for new features

## License

This project is licensed under the MIT License - see the LICENSE file for details.