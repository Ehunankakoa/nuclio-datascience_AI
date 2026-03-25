# %%
import pandas as pd
from openai import OpenAI
from pydantic import BaseModel
import instructor
from enum import Enum
import time

# Remember to use ollama from the app not from the terminal
# Version in the terminal is not working, I don´t know why 


df_raw = pd.read_csv('reviews_Musical_Instruments_sample.csv')

df_raw.head(10)

# %%

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)

df_raw = df_raw[['reviewerID','reviewText','overall']]
df_raw['overall'].value_counts()
df_raw[df_raw['overall']=='1']
 
df_raw1 = df_raw[df_raw['overall']==1].sample(1)
df_raw2 = df_raw[df_raw['overall']==2].sample(1)
df_raw3 = df_raw[df_raw['overall']==3].sample(1)
df_raw4 = df_raw[df_raw['overall']==4].sample(1)
df_raw5 = df_raw[df_raw['overall']==5].sample(1)


df = pd.concat([df_raw1, df_raw2, df_raw3, df_raw4, df_raw5])

df.head(10)
# %%

class sentiment_type(str, Enum):
    positive = 'positive'
    negative = 'negative'
    neutral = 'neutral'
    
class review_length(str, Enum):
    short = 'short'
    medium = 'medium'
    long = 'long'

class ExtractReview(BaseModel):
    sentiment: sentiment_type
    score: int
    instrument: str
    review_length: review_length

client = instructor.from_openai(
    OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
    ),
    mode=instructor.Mode.JSON,
)

# %%
"""Ensure you have ollama installed and running

ollama serve
ollama run gemma3:12b

Sometimess gemma models are strange and they don´t always run on all the versions of ollama, ensure you have the latest

Also, you can use another LLM that accept structured outputs like Meta Llama

"""

def extract_review(review: str) -> ExtractReview:

    start_time = time.time()
    resp = client.chat.completions.create(
        model="gemma3:12b",
        messages=[
            {
                "role": "system",
                "content": """Clasify the product and extract the information given an Amazon review.
                The sentiment can be positive, negative or neutral.
                The score is an integer between 1 and 5.
                The instrument is a string if you cannot find an instrument return the item of review.
                """
            },
            {
                "role": "user",
                "content": """review: {review}""".format(review=review),
            }
        ],
        response_model=ExtractReview,
        
    )
    elapsed_time = time.time() - start_time
    print(f"Ollama request took {elapsed_time:.2f} seconds")
    print('Ollama response: ', resp.model_dump_json())
    print('Sentiment: ', resp.sentiment)
    print('Score: ', resp.score)
    print('Instrument: ', resp.instrument)
    print('Review length: ', resp.review_length)


    return resp

df['llm_result'] = df['reviewText'].apply(extract_review)

df['sentiment'] = df['llm_result'].apply(lambda x: x.sentiment).str.split('.').str[0]
df['score'] = df['llm_result'].apply(lambda x: x.score)
df['instrument'] = df['llm_result'].apply(lambda x: x.instrument)
df['length'] = df['llm_result'].apply(lambda x: x.review_length).str.split('.').str[0]

df.drop(axis = 1, inplace = True, columns = ['llm_result','reviewerID'])

df

pd.get_dummies(df['sentiment'])


# %%


df

# %%