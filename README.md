# SENTIMENT ANALYSIS IN REVIEWS ON GLASSDOOR: A study on Information Technology companies in Cuiabá

In today's digital age, online platforms serve as powerful repositories of user-generated content, offering invaluable insights into consumer experiences and sentiments. Among these platforms, Glassdoor stands out as a prominent hub for employee reviews, providing a wealth of feedback on companies from individuals across various industries.

In this project, we embark on a journey to explore and analyze Glassdoor reviews of Information Technology (IT) companies located in Cuiabá, Brazil. Leveraging the vast potential of Natural Language Processing (NLP) techniques, we aim to unearth hidden patterns, sentiments, and trends embedded within these reviews.

Our endeavor is divided into several key phases:

## 1. Data Mining
The Data Mining process, facilitated by [scraper.ipynb](/data_mining/scraper.ipynb), involves the systematic downloading of HTML review pages from Glassdoor for a predefined list of Companies. Due to the absence of a public API for extracting reviews data from Glassdoor, the `Selenium` web automation tool was utilized. This process entails navigating to each review page and downloading its HTML content, with careful consideration given to incorporating delays between requests to circumvent potential blocking mechanisms imposed by Glassdoor.

The collected HTML pages serve as the raw material for subsequent data extraction and analysis steps, enabling the generation of valuable insights into employee sentiments and company reputations.

## 2. Data Preparation

### 2.1 Data Extraction
The Data Extraction process, facilitated by [data_preparation.ipynb](/data_preparation/data_preparation.ipynb), reads the HTML files downloaded in the previous section to extract reviews and transform them into a structured dataset. The resulting dataset contains the following columns:

- **review_id:** Glassdoor Review ID
- **company:** Company Name
- **employee_role:** Employee Role
- **employee_detail:** Details about the employee, including current employment status and duration
- **review_text:** Review Text
- **review_date:** Review Date
- **star_rating:** Star ratings given by the employee to the company
- **sentiment:** The sentiment of the review (1 for positive, 0 for neutral, -1 for negative)

This process enables the organization and analysis of reviews for further insights and decision-making.

### 2.2 Data Conversion
The Data Conversion process utilizes the [data_conversion.ipynb](/data_preparation/data_conversion.ipynb) notebook to predict the sentiment of reviews using the [citizenlab/twitter-xlm-roberta-base-sentiment-finetunned](https://huggingface.co/citizenlab/twitter-xlm-roberta-base-sentiment-finetunned) model.

This classification is crucial for reviews that do not fit into the categories of purely positive or negative. Since Glassdoor only offers these two options for classifying reviews, predicting sentiments ensures that all reviews are appropriately categorized.

### 2.3 Data Merge
The Data Merge process involves merging annotated sentiments with the extracted sentiments. This task is facilitated by the [data_merge.ipynb](/data_preparation/data_merge.ipynb) notebook.

By combining annotated sentiments, manually reviewed using the Sentiment Annotation Tool, with the initially extracted sentiments, the dataset achieves improved accuracy and reliability in sentiment classification.

### 2.4 Sentiment Annotation Tool
In the [Sentiment Annotation Tool](/data_preparation/annotation_tool.py), predictions from the previous section undergo human review. This tool, built with Streamlit, empowers users to:

- View each review alongside its current sentiment (extracted from Glassdoor) and the sentiment classified by [citizenlab/twitter-xlm-roberta-base-sentiment-finetunned](https://huggingface.co/citizenlab/twitter-xlm-roberta-base-sentiment-finetunned).
- Modify the sentiment of each review as needed, classifying them as:
  - Positive
  - Negative
  - Neutral
- Download the updated dataset with corrected sentiment labels.

![Sentiment Annotation Tool Preview](./data_preparation/annotation_tool_preview.png)

## 3. Data Analysis

The [data_analysis.ipynb](/data_analysis/data_analysis.ipynb) notebook offers an in-depth examination of the Glassdoor reviews dataset, utilizing visualizations and statistical summaries to explore the distribution and sentiment of the review texts. Key components include a general overview that presents basic statistics and visualizations of reviews per company.

The analysis delves into review text characteristics, featuring histograms for text length, word counts, and average word length, alongside token frequency analysis using the BERT tokenizer. It also includes n-grams analysis to identify common bigrams and trigrams by sentiment, as well as word clouds to highlight frequent terms across different sentiment categories. Finally, the notebook presents a comprehensive sentiment analysis, illustrating the distribution of sentiments within the dataset and visualizing results post-manual annotation through various graphical tools.

## 4. Train model

The [train.ipynb](/train_model/train.ipynb) notebook is designed to train a sentiment analysis model on the Glassdoor reviews dataset. It provides a comprehensive workflow for training a sentiment analysis model, from data preparation to model evaluation and saving. Key steps in the notebook include:

### 4.1 Data Preparation:
- Loads and preprocesses the dataset, including text cleaning and tokenization.
- Splits the data into training and validation sets.

### 4.2 Model Setup:
- Configures the model architecture, typically using a pre-trained transformer model such as BERT.
- Sets up the training parameters, including learning rate, batch size, and number of epochs.

### 4.3 Training:
- Trains the model on the training set, monitoring performance on the validation set.
- Utilizes techniques such as early stopping and learning rate scheduling to optimize training.


### 4.4 Evaluation:
- Evaluates the trained model on the validation set, calculating metrics such as accuracy, precision, recall, and F1-score.
- Visualizes the training and validation loss over epochs to assess model performance.


### 4.5 Saving the Model:
- Saves the trained model and tokenizer for future inference and deployment.
