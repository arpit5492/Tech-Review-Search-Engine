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
    print(topkdocs)
    

document = "sample.csv"
lucene.initVM(vmargs=['-Djava.awt.headless=true'])
indexCSV('final1/')
retrieve('final1/', 'PlayStation')

