from flask import Flask, render_template, request

app = Flask(__name__)
def generate_index():
    import torch
    import faiss

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    batch_size = 32  # adjust as needed
    input_ids = []
    attention_mask = []

    index = faiss.IndexFlatIP(768)
    print(index.is_trained)

    num_sentences = 0

    model.to(device)
    tokenizer_kwargs = {'max_length': 512, 'truncation': True, 'padding': 'max_length', 'return_tensors': 'pt'}

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        new_tokens = tokenizer.batch_encode_plus(batch, **tokenizer_kwargs)
        new_tokens = {key: tensor.to(device) for key, tensor in new_tokens.items()}
        input_ids.append(new_tokens['input_ids'])
        attention_mask.append(new_tokens['attention_mask'])

        with torch.no_grad():
            outputs = model(new_tokens['input_ids'], new_tokens['attention_mask'])

        embeddings = outputs.last_hidden_state
        attention_mask_batch = new_tokens['attention_mask'].unsqueeze(-1).expand_as(embeddings).float()
        masked_embeddings = embeddings * attention_mask_batch
        summed = torch.sum(masked_embeddings, 1)
        summed_mask = torch.clamp(attention_mask_batch.sum(1), min=1e-9)
        mean_pooled_batch = summed / summed_mask

    # Add the mean-pooled embeddings to the FAISS index
        index.add(mean_pooled_batch.cpu().numpy().astype('float32'))

        num_sentences += len(batch)

        del new_tokens, embeddings, attention_mask_batch, masked_embeddings, summed, summed_mask, mean_pooled_batch

    print(index.ntotal)

# Define the BERT function
def bert(query):
    # !pip3 install transformers
    # !pip3 install faiss-cpu
    # !pip3 install -U scikit-learn scipy matplotlib

    from transformers import AutoTokenizer, AutoModel
    import torch

    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-distilroberta-v1')
    model = AutoModel.from_pretrained('sentence-transformers/all-distilroberta-v1')

    import pandas as pd
    df = pd.read_csv('testing.csv')
    df = df.dropna()
    import csv
    # Open the CSV file
    with open('testing.csv', 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)

        # Loop over each row in the CSV file
        for row in reader:
            # Extract the review text
            review = row['review']

            # Split the review text into words
            words = review.split()

            # Join the first 400 words into a new string
            new_review = ' '.join(words[:400])

            # Trim the new string to the last complete sentence
            if len(words) > 400:
                last_period_index = new_review.rfind('.')
                if last_period_index != -1:
                    new_review = new_review[:last_period_index+1]

            # Update the row with the truncated review text
            row['review'] = new_review



    df['review'] = df['review'].astype(str)
    review_section = df['review']

# Remove meaningless semicolons and periods
    preprocessed_reviews = []
    for review in review_section:
        review = review.replace(';', '') # remove semicolons
        review = review.replace('. ', '.@@@') # temporarily replace valid periods with a unique delimiter
        review = review.replace('.', '') # remove any remaining periods
        review = review.replace('@@@', '. ') # replace the unique delimiter with valid periods
        preprocessed_reviews.append(review)

    sentences= preprocessed_reviews

    # generate_index() //Uncomment this section if u want to generate new index


    #COSINE SIMILARITY PART

    def convert_to_embedding(query):
        tokens={'input_ids' : [], 'attention_mask': []}
        new_tokens = tokenizer.encode_plus(query,max_length=512,truncation=True, padding='max_length', return_tensors='pt')
        tokens['input_ids'].append(new_tokens['input_ids'][0])
        tokens['attention_mask'].append(new_tokens['attention_mask'][0])
        tokens['input_ids']= torch.stack(tokens['input_ids'])
        tokens['attention_mask']= torch.stack(tokens['attention_mask'])
        with torch.no_grad():
            outputs = model(**tokens)
        embeddings = outputs.last_hidden_state
        attention_mask = tokens['attention_mask']
        mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        masked_embeddings = embeddings * mask
        summed = torch.sum(masked_embeddings,1)
        summed_mask = torch.clamp(mask.sum(1), min=1e-9)
        mean_pooled = summed / summed_mask

        return mean_pooled[0] #assuming qeury is a single sentence

    import faiss
    index_loaded = faiss.read_index("testing1.index")

    query_embedding = convert_to_embedding(query)
    D, I = index_loaded.search(query_embedding[None, :], 10)

    # return results
    related_entries = []
    for i, similarity in zip(I[0], D[0]):
        if i >= 0 and i < len(df):
            title = df.iloc[i]['title']
            review = df.iloc[i]['review'][:600] # get first 4 lines of review
            link = df.iloc[i]['link']
            related_entries.append((title, review, link, similarity))
    print(f"Query: {query}")
    print("Related Entries:")
    return related_entries




# Define the Lucene function
def lucene(query):
    import logging, sys
    logging.disable(sys.maxsize)

    #Import libraries

    import lucene
    import os
    from org.apache.lucene.store import MMapDirectory, SimpleFSDirectory, NIOFSDirectory
    from java.nio.file import Paths
    from org.apache.lucene.analysis.standard import StandardAnalyzer
    from org.apache.lucene.document import Document, Field, FieldType
    from org.apache.lucene.queryparser.classic import QueryParser
    from org.apache.lucene.index import FieldInfo, IndexWriter, IndexWriterConfig, IndexOptions, DirectoryReader
    from org.apache.lucene.search import IndexSearcher, BoostQuery, Query
    from org.apache.lucene.search.similarities import BM25Similarity
    import csv

    #Indexing the sample csv file scraped from the web crawler
    def indexCSV(dir):
        if not os.path.exists(dir):
            os.mkdir(dir)
        store = SimpleFSDirectory(Paths.get(dir))
        analyzer = StandardAnalyzer()
        config = IndexWriterConfig(analyzer)
        config.setOpenMode(IndexWriterConfig.OpenMode.CREATE)
        writer = IndexWriter(store, config)

        metaType = FieldType()
        metaType.setStored(True)
        metaType.setTokenized(False)

        contextType = FieldType()
        contextType.setStored(True)
        contextType.setTokenized(True)
        contextType.setIndexOptions(IndexOptions.DOCS_AND_FREQS) #Used DOCS and FREQUENCY FOR INDEX_OPTIONS
        with open(document,'r') as file:
            csv_reader=csv.reader(file)
            for sample in csv_reader:
                title=sample[0]
                review=sample[1]
                link=sample[2]
                doc = Document()
                doc.add(Field('Title', str(title), contextType))
                doc.add(Field('Review', str(review), contextType))
                doc.add(Field('link',str(link),contextType))
                writer.addDocument(doc)
            writer.close()

    #Function to retrieve
    def retrieve(storedir, query):
        searchDir = NIOFSDirectory(Paths.get(storedir))
        searcher = IndexSearcher(DirectoryReader.open(searchDir))
    
        parser = QueryParser('Review', StandardAnalyzer())
        parsed_query = parser.parse(query)

        topDocs = searcher.search(parsed_query, 10).scoreDocs
        topkdocs = []
        for hit in topDocs:
            doc = searcher.doc(hit.doc)
            topkdocs.append({
                "score": hit.score,
                "text": doc.get("Title")
            })
        return topkdocs
    

    document = "testing.csv"
    lucene.initVM(vmargs=['-Djava.awt.headless=true'])
    indexCSV('final1/')
    result_lucene =retrieve('final1/', query)
    return result_lucene




# Home page

@app.route('/')
def home():
    return render_template('home.html')

# Results page
@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    method = request.form['method']
    results = []
    
    if method == 'bert':
        # Use BERT to search
        results = bert(query)
    elif method == 'lucene':
        # Use Lucene to search
        results = lucene(query)
    
    return render_template('search.html', query=query, method=method, results=results)

if __name__ == '__main__':
    app.run()