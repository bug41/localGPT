import sys

import logging
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import click
import torch
from langchain.docstore.document import Document
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from utils import get_embeddings

from InstructorEmbedding import INSTRUCTOR

from transformers import pipeline

#from sklearn.metrics.pairwise import cosine_similarity
#import sklearn.cluster

from constants import (
    CHROMA_SETTINGS,
    DOCUMENT_MAP,
    EMBEDDING_MODEL_NAME,
    INGEST_THREADS,
    PERSIST_DIRECTORY,
    SOURCE_DIRECTORY,
)


def test_function(device_type):
    logging.info(f"test_function 호출 {device_type}")

@click.command()
@click.option(
    "--device_type",
    default="cuda" if torch.cuda.is_available() else "cpu",
    type=click.Choice(
        ["test","cpu","cuda","ipu","xpu","mkldnn","opengl","opencl","ideep","hip","ve","fpga","ort","xla","lazy","vulkan","mps","meta","hpu","mtia",],
    ),
    help="Device to run on. (Default is cuda)",
)


def main(device_type):
    
    logging.info(f"main 함수 호출")
    test_function(device_type)


    #sentiment_classification = pipeline("text-classification", "SamLowe/roberta-base-go_emotions")
    #result1 = sentiment_classification("today is payday!")
    #print(result1)

    logging.info(f"====================================================================================")
    logging.info(f"====================================================================================")
    
    classifier = pipeline("zero-shot-classification", model='C:/web_developer/python/localGPT/models/bart-large-mnli')


    result = classifier("one day I will see the world ",  candidate_labels = ['travel', 'cooking', 'dancing'])
    
    print(result)    

    #sentences_a = [['Represent the Science sentence: ','Parton energy loss in QCD matter'], 
    #            ['Represent the Financial statement: ','The Federal Reserve on Wednesday raised its benchmark interest rate.']]
    #sentences_b = [['Represent the Science sentence: ','The Chiral Phase Transition in Dissipative Dynamics'],
    #            ['Represent the Financial statement: ','The funds rose less than 0.5 per cent on Friday']]
    #embeddings_a = model.encode(sentences_a)
    #embeddings_b = model.encode(sentences_b)
    #similarities = cosine_similarity(embeddings_a,embeddings_b)
    #print(similarities)

    #logging.info(f"===================== hkunlp/instructor-large =====================")
    #model = INSTRUCTOR('hkunlp/instructor-large')
    #sentence = "3D ActionSLAM: wearable person tracking in multi-floor environments"
    #instruction = "Represent the Science title:"
    #embeddings = model.encode([[instruction,sentence]])
    #logging.info(f"===================== embeddings =====================")
    #print(embeddings)
    #logging.info(f"===================== embeddings =====================")

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )

    main()
    #device_type = sys.argv[2]    
    #main(device_type)
