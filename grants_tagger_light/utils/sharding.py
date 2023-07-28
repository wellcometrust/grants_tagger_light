from datasets import IterableDataset
from loguru import logger
import math


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
        shards = [
            dataset.shard(num_shards=self.num_shards, index=index, contiguous=True)
            for index in range(self.num_shards)
        ]

        return IterableDataset.from_generator(
            self.gen_from_shards, gen_kwargs={"_shards": shards}
        )

    @staticmethod
    def calculate_max_steps(training_args, train_dset_size):
        """This is needed when using IterableDatasets,
        as there is no __len__ in advance since the dataset is a
        generator with yield, so it does not know when to end.
            Source: https://discuss.huggingface.co/t/streaming-dataset-into-trainer-does-not-implement-len-max-steps-has-to-be-specified/32893/6

        Example: allMeSH_2021.json has 15.559.157 rows, with 5% for test
                > 15559157-0.05*15559157 = 14781199.15 training rows
                let's suppose batch size is 8
                >  14781199.15 / 8 = 1847649.89375
                if accumulation_steps is 1, then
                > 1847649.89375 / 1 = 1847649.89375
        """  # noqa

        # From https://huggingface.co/transformers/v4.2.2/_modules/transformers/trainer_tf.html
        train_batch_size = training_args.per_device_train_batch_size
        logger.info(f"train_dset_size: {train_dset_size}")
        logger.info(f"train_batch_size: {train_batch_size}")

        num_update_steps_per_epoch = train_dset_size / train_batch_size
        num_update_steps_per_epoch = math.ceil(num_update_steps_per_epoch)
        logger.info(f"num_update_steps_per_epoch: {num_update_steps_per_epoch}")

        accumulation_steps = max(training_args.gradient_accumulation_steps, 1)
        logger.info(f"accumulation steps: {accumulation_steps}")

        num_update_steps_per_epoch /= accumulation_steps
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
        logger.info(f"num_update_steps_per_epoch: {num_update_steps_per_epoch}")

        epochs = max(training_args.num_train_epochs, 1)
        logger.info(f"epochs: {epochs}")

        gpus = max(training_args._n_gpu, 1)
        logger.info(f"gpus using HF: {gpus}")

        total_steps = num_update_steps_per_epoch * epochs
        logger.info(f"total_steps: {total_steps}")

        total_steps_per_gpu = math.ceil(total_steps / gpus)
        logger.info(f"total_steps_per_gpu: {total_steps_per_gpu}")
        return total_steps_per_gpu
