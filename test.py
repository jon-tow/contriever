import os
import torch
import logging
import train
import torch.distributed as dist
import src.slurm as slurm
import src.utils as utils
import src.dist_utils as dist_utils
import src.moco as moco
import src.data as data
from torch.utils.data import DataLoader, RandomSampler
from src.options import Options 

logger = logging.getLogger(__name__)

def train(opt, model, optimizer, scheduler, step, wandb_run = None):
    logger.info("Data loading")
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        tokenizer = model.module.tokenizer
    else:
        tokenizer = model.tokenizer
    collator = data.Collator(opt=opt)
    train_dataset = data.load_data(opt, tokenizer)
    logger.warning(f"Data loading finished for rank {dist_utils.get_rank()}")
    
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, 
        sampler=train_sampler, 
        batch_size=opt.per_gpu_batch_size, 
        drop_last=True, 
        num_workers=opt.num_workers, 
        collate_fn=collator
    )

    epoch = 1

    model.train()
    while step < opt.total_steps:
        train_dataset.generate_offset()

        logger.info(f'Start epoch {epoch}')
        for i, batch in enumerate(train_dataloader):
            step += 1

            # TODO (jon-tow): Don't put full batches on GPU yet (it's slow)
            batch = {
                key: value.cuda() 
                if isinstance(value, torch.Tensor) else value 
                for key, value in batch.items()
            }
            train_loss, _ = model.forward(**batch, stats_prefix='train')
            
            # train_loss.backward()
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            raise RuntimeError("Stop")


if __name__ == "__main__":

    options = Options()
    opt = options.parse()

    torch.manual_seed(opt.seed)
    slurm.init_distributed_mode(opt)
    slurm.init_signal_handler()


    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    model_class = moco.MoCo
    model = model_class(opt)
    model = model.cuda()
    optimizer, scheduler = utils.set_optim(opt, model)
    step = 0
    logger.info(utils.get_parameters(model))

    # Turn this off while testing
    if dist.is_initialized():
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            find_unused_parameters=True,
        )
        dist.barrier()
  
    logger.info("Start training")
    train(opt, model, optimizer, scheduler, step, None)
