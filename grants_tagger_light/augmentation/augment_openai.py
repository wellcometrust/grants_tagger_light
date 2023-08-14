import json
import os
from loguru import logger
import openai


class AugmentOpenAI:
    def __init__(self, prompt_template_path, model_key='gpt-3.5-turbo'):
        if 'OPENAI_API_KEY' not in os.environ:
            logger.error("OPENAI_API_KEY not found in env vars. Please define it before running this program.")
        with open(prompt_template_path, 'r') as f:
            self.prompt_template = f.read()
        self.model_key = model_key

    def generate(self, featured_tag, dataset, n=1, few_shot_examples=10, temperature=1.5, top_p=1, frequence_penalty=0,
                 presence_penalty=0):
        size = min(len(dataset), few_shot_examples)
        dataset = dataset[:size]
        abstracts = "\n".join(dataset['abstractText'])
        tags = []
        for x in dataset['meshMajor']:
            tags.extend(x)
        mesh_tags = ",".join(list(set(tags)))

        prompt = self.prompt_template.replace('{FEATURED_TAG}', featured_tag)
        prompt = prompt.replace('{ABSTRACTS}', abstracts)
        prompt = prompt.replace('{MESH_TAGS}', mesh_tags)

        response = openai.ChatCompletion.create(
            model=self.model_key,
            messages=[
                {"role": "user", "content": prompt}],
            n=n,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequence_penalty,
            presence_penalty=presence_penalty
        )
        for r in response['choices']:
            if 'message' in r:
                if 'content' in r['message']:
                    print(r['message']['content'])
                    try:
                        r_json = json.loads(r['message']['content'])
                        a = r_json['abstract']
                        # Make sure it does not hallucinate and adds anything new which may not be a MeSH tag
                        t = [x for x in r_json['tags'] if x in tags]
                        tl = r_json['title']
                        yield {'abstract': a, 'tags': t, 'title': tl}
                    except Exception as e:
                        logger.info("OpenAI did not return a proper json format.")
                        yield None
