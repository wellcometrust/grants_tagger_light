from transformers import Trainer, AdamW, get_cosine_schedule_with_warmup
from loguru import logger

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def create_optimizer_and_scheduler(self, num_training_steps):
        # Instantiate the AdamW optimizer with a constant learning rate
        self.optimizer = AdamW(self.model.parameters(),
                               lr=self.args.learning_rate)  # Set your desired learning rate
        logger.info(f"Optimizer: {self.optimizer}")
        # Create a learning rate scheduler
        self.lr_scheduler = get_cosine_schedule_with_warmup(self.optimizer,
                                                            num_warmup_steps=self.args.warmup_steps,
                                                            num_training_steps=self.args.max_steps)
        logger.info(f"Scheduler: {self.lr_scheduler}")
