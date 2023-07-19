from datasets import load_dataset, IterableDataset


class Sharding:
    def __init__(self, num_shards=100):
        """
        Sharding to prevent processing time issues, suggested here https://github.com/huggingface/datasets/issues/2252
        Args:
            num_shards: num of shards to split the train dataset into.
        """
        self.num_shards = num_shards

    @classmethod
    def gen_from_shards(cls, _shards):
        for shard in _shards:
            for example in shard:
                yield example

    def shard(self, dataset):
        shards = [dataset.shard(num_shards=self.num_shards, index=index, contiguous=True)
                  for index in range(self.num_shards)]

        return IterableDataset.from_generator(self.gen_from_shards, gen_kwargs={"_shards": shards})
