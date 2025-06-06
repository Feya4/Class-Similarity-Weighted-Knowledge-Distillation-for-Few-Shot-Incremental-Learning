import numpy as np
import torch
from dataloader.sampler import CategoriesSampler
from dataloader.samplers import Batch_MetaSampler

# def set_up_datasets(args):
#     if args.dataset == 'cifar100':
#         import dataloader.cifar100.cifar as Dataset
#         args.base_class = 60
#         args.num_classes=100
#         args.way = 5
#         args.shot = 5
#         args.sessions = 9
#     if args.dataset == 'cub200':
#         import dataloader.cub200.cub200 as Dataset
#         args.base_class = 100
#         args.num_classes = 200
#         args.way = 10
#         args.shot = 5
#         args.sessions = 11
#     if args.dataset == 'mini_imagenet':
#         import dataloader.miniimagenet.miniimagenet as Dataset
#         args.base_class = 60
#         args.num_classes=100
#         args.way = 5
#         args.shot = 5
#         args.sessions = 9
#     return args


#def set_up_datasets(args):
def dataset_up_sets(args):
    if 'cifar' in args.dataset.lower():
        import dataloader.cifar100.cifar as Dataset
        args.dataset = 'cifar100'
        args.base_class = 60
        args.num_classes = 100
        args.n_way = 5
        args.n_shot = 5
        args.n_sessions = 9
    if 'cub' in args.dataset.lower():
        import dataloader.cub200.cub200 as Dataset
        args.dataset = 'cub200'
        args.base_class = 100
        args.num_classes = 200
        args.n_way = 10
        args.n_shot = 5
        args.n_sessions = 11
    if 'mini' in args.dataset.lower():
        import dataloader.miniimagenet.miniimagenet as Dataset
        args.dataset = 'mini_imagenet'
        args.base_class = 60
        args.num_classes = 100
        args.n_way = 5
        args.n_shot = 1
        args.n_sessions = 9
    else:
        Exception('Undefined dataset name!')
    args.Dataset = Dataset
    return args


def get_dataloader(args,session):
    if session == 0:
        trainset, trainloader, testloader = get_base_dataloader(args)
    else:
        trainset, trainloader, testloader = get_new_dataloader(args)
    return trainset, trainloader, testloader


def get_dataloader_base(args):
    txt_path = "./FeyaFSCIL/data/index_list/" + args.dataset + "/session_" + str(0 + 1) + '.txt'
    class_index = np.arange(args.base_class)
    if args.dataset == 'cifar100':

        trainset = args.Dataset.CIFAR100(root=args.dataroot, train=True, download=False,
                                         index=class_index, base_sess=True)
        testset = args.Dataset.CIFAR100(root=args.dataroot, train=False, download=False,
                                        index=class_index, base_sess=True)

    if args.dataset == 'cub200':
        trainset = args.Dataset.CUB200(root=args.dataroot, train=True,
                                       index=class_index, base_sess=True)
        testset = args.Dataset.CUB200(root=args.dataroot, train=False, index=class_index)

    if args.dataset == 'mini_imagenet':
        trainset = args.Dataset.MiniImageNet(root=args.dataroot, train=True,
                                             index=class_index, base_sess=True)
        testset = args.Dataset.MiniImageNet(root=args.dataroot, train=False, index=class_index)


    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True,
                                              num_workers=0, pin_memory=True)
    testloader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=args.batch_size_test, shuffle=False, num_workers=8, pin_memory=True)

    return trainset, trainloader, testloader


def get_dataloader_new(args):
    txt_path = "./FeyaFSCIL/data/index_list/" + args.dataset + "/session_" + str(0 + 1) + '.txt'
    class_index = np.arange(args.base_class)
    if args.dataset == 'cifar100':
        trainset = args.Dataset.CIFAR100(root=args.dataroot, train=True, download=True,
                                         index=class_index, base_sess=True)
        testset = args.Dataset.CIFAR100(root=args.dataroot, train=False, download=False,
                                        index=class_index, base_sess=True)

    if args.dataset == 'cub200':
        trainset = args.Dataset.CUB200(root=args.dataroot, train=True,
                                       index_path=txt_path)
        testset = args.Dataset.CUB200(root=args.dataroot, train=False,
                                      index=class_index)
    if args.dataset == 'mini_imagenet':
        trainset = args.Dataset.MiniImageNet(root=args.dataroot, train=True,
                                             index_path=txt_path)
        testset = args.Dataset.MiniImageNet(root=args.dataroot, train=False,
                                            index=class_index)

    # DataLoader(test_set, batch_sampler=sampler, num_workers=8, pin_memory=True)
    sampler = Batch_MetaSampler(trainset.targets, args.n_episode_train, args.n_way_train,
                                args.n_shot_train, args.n_query_train, args.task_per_batch)

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_sampler=sampler, num_workers=args.num_workers,
                                              pin_memory=True)

    testloader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=args.batch_size_test, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    return trainset, trainloader, testloader


def get_new_dataloader(args,session):
    txt_path = "./FeyaFSCIL/data/index_list/" + args.dataset + "/session_" + str(session + 1) + '.txt'
    if args.dataset == 'cifar100':
        class_index = open(txt_path).read().splitlines()
        trainset = args.Dataset.CIFAR100(root=args.dataroot, train=True, download=False,
                                         index=class_index, base_sess=False)
    if args.dataset == 'cub200':
        trainset = args.Dataset.CUB200(root=args.dataroot, train=True,
                                       index_path=txt_path)
    if args.dataset == 'mini_imagenet':
        trainset = args.Dataset.MiniImageNet(root=args.dataroot, train=True,
                                       index_path=txt_path)
    if args.batch_size_new == 0:
        batch_size_new = trainset.__len__()
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size_new, shuffle=False, # shuffle=True
                                                  num_workers=args.num_workers, pin_memory=True)
    else:
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=32, shuffle=True,
                                                  num_workers=args.num_workers, pin_memory=True)#batch_size=args.batch_size 'FOR IMAGENET AND CUB200'

    # test on all encountered classes
    class_new = get_session_classes(args, session)

    if args.dataset == 'cifar100':
        testset = args.Dataset.CIFAR100(root=args.dataroot, train=False, download=False,
                                        index=class_new, base_sess=False)
    if args.dataset == 'cub200':
        testset = args.Dataset.CUB200(root=args.dataroot, train=False,
                                      index=class_new)
    if args.dataset == 'mini_imagenet':
        testset = args.Dataset.MiniImageNet(root=args.dataroot, train=False,
                                      index=class_new)

    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=args.batch_size_test, shuffle=False,
                                             num_workers=args.num_workers, pin_memory=True)

    return trainset, trainloader, testloader


def get_session_classes(args,session):
    class_list = np.arange(args.base_class + session * args.n_way)
    return class_list
