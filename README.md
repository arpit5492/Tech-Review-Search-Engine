# TechreviewSearchEngine

This project implements a search engine for text data using two different methods: BERT embeddings and Lucene indexing. The search engine allows users to input a query and select a search method (BERT or Lucene) to retrieve relevant documents from a dataset. The dataset used for indexing and retrieval is a sample CSV file containing title and review information.

## Features
Utilizes the BERT model for generating embeddings of input queries and documents.
Implements Lucene indexing for efficient and fast retrieval of documents based on search queries.
Provides a web-based user interface to input queries and choose the search method.
Displays top search results along with scores and text snippets.


## Getting Started
Clone this repository: git clone https://github.com/your-username/search-engine.git
Install the required Python packages: pip install -r requirements.txt
Download and install Faiss for efficient similarity search.
Download and install PyLucene for Lucene indexing.
Download and install the necessary Python libraries for BERT: pip install transformers sentence-transformers pandas
Prepare your data: Create a CSV file containing title and review information.
Update the app.py file with the appropriate paths to your data and indexes.
Run the application: python app.py
Access the application in your web browser at http://localhost:5000.

## Usage
Open the application in your web browser.
Input your search query and select the search method (BERT or Lucene) from the dropdown menu.
Click the "Search" button.
View the search results displayed on the results page.
Click on a result to view more details.
