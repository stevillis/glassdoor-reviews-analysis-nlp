# Glassdoor Reviews Analysis of IT Companies in Cuiabá with Natural Language Processing

## Data Mining
Uses [scraper.ipynb](/data_mining/scraper.ipynb) to download the `html` reviews pages from Glassdoor for a pre-select list of Companies. Since Glassdoor has no public available API to extract reviews data, `Selenium` was used to download each page with some delay between requests to bypass blocking from Glassdoor.

## Data Preparation

### Data Extraction
Uses [data_preparation.ipynb](/data_preparation/data_preparation.ipynb)  reads the `html` files downloaded in the previous section to collect reviews and transform them into a dataset with the columns:
- review_id: Glassdoor Review ID
- company: Company Name
- employee_role: Employee Role
- employee_detail: Details about the employee, including whether they are currently employed by the company or not, and for how long
- review_text: Review Text
- review_date: Review Date
- star_rating: The star ratings that the employee gave to the company
- sentiment: The sentiment of the review (0 for contrary and 1 for positive)

### Data Conversion
Uses the [data_conversion.ipynb](/data_preparation/data_conversion.ipynb) to predict sentiment of reviews with the [citizenlab/twitter-xlm-roberta-base-sentiment-finetunned](https://huggingface.co/citizenlab/twitter-xlm-roberta-base-sentiment-finetunned) model. This classification is necessary for reviews that are neither purely positive nor negative, as Glassdoor only allows these two options for classifying reviews.

### Sentiment Annotation Tool
The predictions made in the previous section are reviewed by a human using the [Sentiment Annotation Tool](/data_preparation/annotation_tool.py), created with `Streamlit`. This tool allows the user to:
- View each review, its current sentiment (extracted from Glassdoor), and the sentiment classified by  [citizenlab/twitter-xlm-roberta-base-sentiment-finetunned](https://huggingface.co/citizenlab/twitter-xlm-roberta-base-sentiment-finetunned)
- Modify the sentiment of each review if necessary, by classifying them as:
  - Positive
  - Negative
  - Neutral
- Download the new dataset with fixed sentiment classes

![Sentiment Annotation Tool Preview](./data_preparation/annotation_tool_preview.png)
