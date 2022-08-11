import builtins
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

            optimizer.zero_grad()

            # TODO (jon-tow): Don't put full batches on GPU yet (it's slow)
            batch = {
                key: value.cuda() 
                if isinstance(value, torch.Tensor) else value 
                for key, value in batch.items()
            }
            with torch.cuda.amp.autocast(True):
                _, iter_stats = model(**batch, stats_prefix='train')

            optimizer.step()
            scheduler.step()


if __name__ == "__main__":

    options = Options()
    opt = options.parse()

    torch.manual_seed(opt.seed)
    # Use `torchrun` manually
    # slurm.init_distributed_mode(opt)
    slurm.init_signal_handler()

    # Read environment variables when using torchrun
    global_rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ["LOCAL_RANK"])
    opt.local_rank = local_rank

    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        init_method='env://',
        backend='nccl',
        world_size=world_size,
        rank=global_rank,
    )
    # Set GPU device
    if dist.is_initialized():
        dist.barrier(device_ids=[local_rank])

    # Suppress printing if not on master gpu
    if global_rank != 0:
        import builtins
        def print_pass(*args):
            pass
        builtins.print = print_pass

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    model_class = moco.MoCo
    model = model_class(opt)
    model = model.cuda()
    # Only optimize the query encoder via the `optimizer` 
    optimizer, scheduler = utils.set_optim(opt, model.encoder_q)
    step = 0
    logger.info(utils.get_parameters(model))
    logger.info("Start training")
    train(opt, model, optimizer, scheduler, step, None)
    dist.destroy_process_group()
