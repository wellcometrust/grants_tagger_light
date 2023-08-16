import json
import os
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

    def _create_message(self, featured_tag, featured_tag_dset):
        abstracts = "\n".join(featured_tag_dset['abstractText'])
        tags = []
        for x in featured_tag_dset['meshMajor']:
            tags.extend(x)

        mesh_tags = ",".join(list(set(tags)))

        prompt = self.prompt_template.replace('{FEATURED_TAG}', featured_tag)
        prompt = prompt.replace('{ABSTRACTS}', abstracts)
        prompt = prompt.replace('{MESH_TAGS}', mesh_tags)

        return [{"role": "user", "content": prompt}]

    def _make_requests(self,
                       collect_concurrent_calls,
                       dset,
                       few_shot_examples,
                       temperature,
                       top_p,
                       presence_penalty,
                       num_proc):
        for num in range(len(collect_concurrent_calls)):
            t = collect_concurrent_calls[num][0]
            n = collect_concurrent_calls[num][1]
            # RAG: I select similar articles to provide them to the LLM
            tmp_dset = dset.filter(lambda x: any(np.isin([t], x["meshMajor"])), num_proc=num_proc)
            # I remove them from the dataset to process to make it smaller and quicker over time
            dset = dset.filter(lambda example, idx: idx not in tmp_dset['idx'], with_indices=True,
                               num_proc=num_proc)
            size = min(len(tmp_dset), few_shot_examples)
            tmp_dset = tmp_dset[:size]

            self.api.request(data={
                "model": self.model_key,
                "n": n,
                "temperature": temperature,
                "top_p": top_p,
                "presence_penalty": presence_penalty,
                "messages": self._create_message(t, tmp_dset)
            }, metadata={'num': num})

    def generate(self, collect_concurrent_calls, dset, few_shot_examples=10, temperature=1.5, top_p=1,
                 presence_penalty=0, num_proc=os.cpu_count()):

        self.api.run_request_function(self._make_requests,
                                      collect_concurrent_calls=collect_concurrent_calls,
                                      dset=dset,
                                      few_shot_examples=few_shot_examples,
                                      temperature=temperature,
                                      top_p=top_p,
                                      presence_penalty=presence_penalty,
                                      num_proc=num_proc)

        for result in self.api:
            num = result.metadata['num']
            for r in result.response['choices']:
                if 'message' in r:
                    if 'content' in r['message']:
                        try:
                            r_json = json.loads(r['message']['content'])
                            a = r_json['abstract']
                            # Make sure it does not hallucinate and adds anything new which may not be a MeSH tag
                            t = [x for x in r_json['tags'] if x in tags]
                            tl = r_json['title']
                            print("YIELD!!!!!!!!!!!!!!")
                            yield {'abstract': a, 'tags': t, 'title': tl}
                        except Exception as e:

                            print("ERROR!!!!!!!!!!!!!!")
                            logger.info("OpenAI did not return a proper json format.")
                            yield None

