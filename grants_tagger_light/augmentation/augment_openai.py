import json
import os
import random
import uuid

from loguru import logger
import openai
from openai_multi_client import OpenAIMultiClient
import numpy as np


class AugmentOpenAI:
    def __init__(self, prompt_template_path, model_key='gpt-3.5-turbo'):
        if 'OPENAI_API_KEY' not in os.environ:
            logger.error("OPENAI_API_KEY not found in env vars. Please define it before running this program.")
        with open(prompt_template_path, 'r') as f:
            self.prompt_template = f.read()
        self.model_key = model_key
        self.api = OpenAIMultiClient(endpoint="chats", data_template={"model": self.model_key})


    @staticmethod
    def _get_secondary_tags(featured_tag_dset):
        tags = []
        for x in featured_tag_dset['meshMajor']:
            tags.extend(x)
        return ",".join(list(set(tags)))

    def _create_message(self, abstract, tag):
        prompt = self.prompt_template.replace('{TOPIC}', tag)
        prompt = prompt.replace('{ABSTRACT}', abstract)

        return [{"role": "user", "content": prompt}]

    @staticmethod
    def _process_response(result):
        print("Response!!!")
        with open('kk.kk', 'w') as f:
            f.write(str(result))
        print(result)
        if result.failed:
            logger.warning(f"Failed to get augmentation for {result.metadata['featured_tag']}")
            return

        with open(result.metadata['save_to_path'], 'w') as f:

            for r in result.response['choices']:
                if 'message' in r:
                    if 'content' in r['message']:
                        try:
                            pieces = json.loads(r['message']['content'])
                            a = pieces['abstract']
                            tl = pieces['title']

                            f.write(json.dumps({
                                "journal": result.metadata['model_key'],
                                "meshMajor": result.metadata['tags'],
                                "year": [
                                    result.metadata['year']
                                ],
                                "abstractText": a,
                                "pmid": uuid.uuid4().hex,
                                "title": tl,
                                "existing_example": result.metadata['example'],
                                "required_examples": result.metadata['required_examples'],
                                "featured_tag": result.metadata['featured_tag']
                            }))
                            f.write('\n')
                            f.flush()
                        except Exception as e:
                            logger.info("OpenAI did not return a proper json format...")

    def _make_requests(self,
                       collect_concurrent_calls,
                       dset,
                       few_shot_examples,
                       temperature,
                       top_p,
                       presence_penalty,
                       num_proc,
                       year,
                       model_key,
                       save_to_path):
        for num in range(len(collect_concurrent_calls)):
            t = collect_concurrent_calls[num][0]
            n = collect_concurrent_calls[num][1]
            logger.info(f"Augmenting {t} with {n} examples")
            # RAG: I select similar articles to provide them to the LLM
            tmp_dset = dset.filter(lambda x: any(np.isin([t], x["meshMajor"])), num_proc=num_proc)
            # I remove them from the dataset to process to make it smaller and quicker over time
            dset = dset.filter(lambda example, idx: idx not in tmp_dset['idx'], with_indices=True,
                               num_proc=num_proc)

            for i in range(n):
                selected_row = random.randint(0, len(tmp_dset)-1)
                print(f"Selected row: {selected_row}")
                print(f"Size of dataset: {len(tmp_dset['abstractText'])}")
                abstract = tmp_dset['abstractText'][selected_row]
                tags = tmp_dset['meshMajor'][selected_row]
                data = {
                    "model": self.model_key,
                    "n": n,
                    "temperature": temperature,
                    "top_p": top_p,
                    "presence_penalty": presence_penalty,
                    "messages": self._create_message(abstract, t)
                }

                metadata = {
                    'featured_tag': t,
                    'tags': tags,
                    'required_examples': n,
                    'existing_examples_used': size,
                    'existing_example': abstract,
                    'year': year,
                    'model_key': model_key,
                    'save_to_path': save_to_path
                }

                print(data)
                print(metadata)
                self.api.request(data=data, metadata=metadata, callback=self._process_response)

    def generate(self, collect_concurrent_calls, dset, save_to_path, year, model_key, few_shot_examples=10,
                 temperature=1.5, top_p=1, presence_penalty=0, num_proc=os.cpu_count()):

        self.api.run_request_function(self._make_requests,
                                      collect_concurrent_calls=collect_concurrent_calls,
                                      dset=dset,
                                      few_shot_examples=few_shot_examples,
                                      temperature=temperature,
                                      top_p=top_p,
                                      presence_penalty=presence_penalty,
                                      num_proc=num_proc,
                                      year=year,
                                      model_key=model_key,
                                      save_to_path=save_to_path)

