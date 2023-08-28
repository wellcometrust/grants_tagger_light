import datetime
import json
import os
import random
import uuid

from loguru import logger
import openai
from openai_multi_client import OpenAIMultiClient
import numpy as np

from grants_tagger_light.augmentation.JsonParser import JsonParser


class AugmentOpenAI:
    def __init__(self, prompt_template_path, model_key='gpt-3.5-turbo'):
        if 'OPENAI_API_KEY' not in os.environ:
            logger.error("OPENAI_API_KEY not found in env vars. Please define it before running this program.")
        with open(prompt_template_path, 'r') as f:
            self.prompt_template = f.read()
        self.model_key = model_key
        self.api = OpenAIMultiClient(endpoint="chats", data_template={"model": self.model_key})

    def _create_message(self, abstract, tag):
        prompt = self.prompt_template.replace('{TOPIC}', tag)
        prompt = prompt.replace('{ABSTRACT}', abstract)

        return [{"role": "user", "content": prompt}]

    @staticmethod
    def _process_response(result):
        if result.failed:
            logger.warning(f"Failed to get augmentation for {result.metadata['featured_tag']}")
            return

        with open(result.metadata['save_to_path'], 'a') as f:
            for r in result.response['choices']:
                if 'message' in r:
                    if 'content' in r['message']:
                        try:
                            json_response = JsonParser.parse_json(r['message']['content'])
                        except Exception as e:
                            logger.info(f"Error processing output: {e}. Skipping...")
                            continue

                        a = json_response['abstract']
                        tl = json_response['title']

                        f.write(json.dumps({
                            "journal": result.metadata['model_key'],
                            "meshMajor": result.metadata['tags'],
                            "year": result.metadata['year'],
                            "abstractText": a,
                            "pmid": uuid.uuid4().hex,
                            "title": tl,
                            "existing_example": result.metadata['existing_example'],
                            "required_examples": result.metadata['required_examples'],
                            "featured_tag": result.metadata['featured_tag']
                        }))
                        f.write('\n')
                        f.flush()

                        logger.info(f"Data received successfully for {result.metadata['featured_tag']}")

    def _make_requests(self,
                       collect_concurrent_calls,
                       dset,
                       temperature,
                       top_p,
                       presence_penalty,
                       num_proc,
                       model_key,
                       save_to_path):

        year = datetime.date.year

        for num in range(len(collect_concurrent_calls)):
            t = collect_concurrent_calls[num][0]
            n = collect_concurrent_calls[num][1]
            logger.info(f"Augmenting {t} with {n} examples")
            # RAG: I select similar articles to provide them to the LLM
            tmp_dset = dset.filter(lambda x: any(np.isin([t], x["meshMajor"])), num_proc=num_proc)
            # I remove them from the dataset to process to make it smaller and quicker over time
            dset = dset.filter(lambda example, idx: idx not in tmp_dset['idx'], with_indices=True,
                               num_proc=num_proc)

            abstracts_num = [i for i in range(len(tmp_dset))]
            random.shuffle(abstracts_num)

            for i in range(n):
                selected_row = abstracts_num[i % len(tmp_dset)]
                abstract = tmp_dset['abstractText'][selected_row]
                tags = tmp_dset['meshMajor'][selected_row]
                data = {
                    "model": self.model_key,
                    "n": 1,
                    "temperature": temperature,
                    "top_p": top_p,
                    "presence_penalty": presence_penalty,
                    "messages": self._create_message(abstract, t)
                }

                metadata = {
                    'featured_tag': t,
                    'tags': tags,
                    'required_examples': n,
                    'existing_example': abstract,
                    'year': year,
                    'model_key': model_key,
                    'save_to_path': save_to_path
                }

                self.api.request(data=data, metadata=metadata, callback=self._process_response)

    def generate(self, collect_concurrent_calls, dset, save_to_path, model_key,
                 temperature=1.5, top_p=1, presence_penalty=0, num_proc=os.cpu_count()):
        self.api.run_request_function(self._make_requests,
                                      collect_concurrent_calls=collect_concurrent_calls,
                                      dset=dset,
                                      temperature=temperature,
                                      top_p=top_p,
                                      presence_penalty=presence_penalty,
                                      num_proc=num_proc,
                                      model_key=model_key,
                                      save_to_path=save_to_path)

        self.api.pull_all()
