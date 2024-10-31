from litellm import completion, encode
import os
import pandas as pd
import re
from evaluate import load
from tqdm.auto import tqdm
from litellm.litellm.utils import TextCompletionResponse


def litellm_generate_call(prompt, model_name):

    os.environ['ANYSCALE_API_KEY'] = ".................."
    response = completion(
        model=model_name, 
        messages=prompt, temperature=0.8, top_p=0.9, stream=False, max_tokens=1000
    )
    return response

def generate_query_small():
    query_input_en = "Mr. Reagan, in his first term, tried to kill the agency"
    query_input_de = "Die Weichen für diesen harten aber unerwarteten Ritt hatten die USA höchstselbst gestellt , durch ihre irrationale und ebenso völkerrechtswidrige Sanktionspolitik gegenüber Russland."
    query_input_fr =  "En s'en prenant à ce groupe plutôt qu'aux problèmes, les dirigeants des groupes ont manqué une belle occasion"

    sample_input = "I have a beef with my homies"
    sample_output = ['I', 'have', 'a beef']

    prompt = f"""
        ###Instruction###:
        Generate a LIST that contain the Metaphorical Subject-Verb-Object tuple extracted from the given sentence: {query_input_en}

        ###Example###:
        The following is just a sample data with its output, containing sample_output for the sample_input. \n sample_input:{sample_input} and sample_output:{sample_output} \n

        ###Content###:
        Return an EMPTY LIST if the sentence has Literal meaning. \n
        The VERB must be lemmatized. \n
        """

    messages = [
        { "content":prompt,"role": "user"},
        ]

    response = litellm_generate_call(messages)
    print(response)

def generate_queries(model_name, sampled_query):

    sample_input = "I have a beef with my homies"
    sample_output = ['I', 'have', 'a beef']


    outputs = {}
    outputs['input'] = []
    outputs['output'] = []
    for index, row in tqdm(sampled_query.iterrows()):

        query_input = str(row['Input'])
        outputs['input'].append(query_input)

        prompt = f"""
            ###Instruction###:
            Generate a LIST that contain the Metaphorical Subject-Verb-Object tuple extracted from the given sentence: {query_input}

            ###Example###:
            The following is just a sample data with its output, containing sample_output for the sample_input. \n sample_input:{sample_input} and sample_output:{sample_output} \n

            ###Content###:
            Return an EMPTY LIST if the sentence has Literal meaning. \n
            The VERB must be lemmatized. \n
            """

        messages = [
            { "content":prompt,"role": "user"},
        ]

        response = litellm_generate_call(messages, model_name=model_name)
        outputs['output'].append(response)

    return outputs


sampled_query = pd.read_csv('/home/lyoko/Documents/Personal_Project/Metaphors-Detection/data/en/sample_50.csv')

model_name_1 = "anyscale/meta-llama/Meta-Llama-3-70B-Instruct"
model_name_2 = "anyscale/openai/gpt-4o-turbo-0301"

generate_queries(model_name=model_name_1, sampled_query=sampled_query)
