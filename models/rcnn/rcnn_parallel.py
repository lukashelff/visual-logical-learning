import os
import re
import time
from collections import OrderedDict

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from rtpt.rtpt import RTPT
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler

import models.rcnn.engine as engine
from models.rcnn.inference import infer_symbolic


def rcnn_distributed_training(out_path, model, dl, optimizer, scheduler, rank, world_size, num_epochs=25,
                              save_model=True, rtpt_extra=0, ex_name=None, ):
    ex_name = f'mask_rcnn_perception' if ex_name is None else ex_name
    rtpt = RTPT(name_initials='LH', experiment_name=ex_name, max_iterations=num_epochs + rtpt_extra)
    rtpt.start()
    epoch_init = 0
    print(f'settings: {out_path}')

    # setup the process groups
    setup(rank, world_size)  # prepare the dataloader
    train_loader = prepare_dl(rank, world_size, dl['train'])
    val_loader = prepare_dl(rank, world_size, dl['val'])

    # instantiate the model(it's your own model) and move it to the right device
    model = model.to(rank)

    # wrap the model with DDP
    # device_ids tell DDP where is your model
    # output_device tells DDP where to output, in our case, it is rank
    # find_unused_parameters=True instructs DDP to find unused output of the forward() function of any module in the model
    model = DistributedDataParallel(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    device = f'cuda:{rank}' if torch.cuda.is_available() else 'cpu'
    for epoch in range(num_epochs):
        rtpt.step()

        train_loader.sampler.set_epoch(epoch)

        print(f"EPOCH {epoch + 1} of {num_epochs}")

        # start timer and carry out training and validation
        start = time.time()

        # train for one epoch, printing every 10 iterations
        print('Training')
        engine.train_one_epoch(model, optimizer, train_loader, device, scheduler, epoch, print_freq=100)
        # update the learning rate
        print('Inferring symbolic representation')
        val_loader.sampler.set_epoch(epoch)
        _, _, acc, mean = infer_symbolic(model, val_loader, device=device, debug=False)

        end = time.time()
        print(f"Took {((end - start) / 60):.3f} minutes for epoch {epoch}")

    os.makedirs(out_path, exist_ok=True)
    if save_model:
        torch.save({
            'epoch': num_epochs + epoch_init,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, out_path + 'model.pth')


def train_parallel(out_path, model, dl, optimizer, scheduler, rank, num_epochs, save_model, rtpt_extra, ex_name,
                   world_size=3, ):
    mp.spawn(
        rcnn_distributed_training,
        args=(
            out_path, model, dl, optimizer, scheduler, rank, world_size, num_epochs, save_model, rtpt_extra, ex_name),
        nprocs=world_size
    )


def load_model(state_dict_model, state_dict):
    # in case we load a DDP model checkpoint to a non-DDP model
    model_dict = OrderedDict()
    pattern = re.compile('module.')
    for k, v in state_dict.items():
        if re.search("module", k):
            model_dict[re.sub(pattern, '', k)] = v
        else:
            model_dict = state_dict_model.load_state_dict(model_dict)


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def prepare_dl(rank, world_size, dataloader, pin_memory=False, num_workers=0):
    sampler = DistributedSampler(dataloader.dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    dataloader.sampler = sampler
    dataloader.drop_last = True
    # dataloader.batch_size = batch_size
    dataloader.num_workers = num_workers
    dataloader.pin_memory = pin_memory
    return dataloader

