from os.path import join

import torch
from torch.utils.data import DataLoader
import numpy as np
from scipy import io
from data.random_dataset import RandDataset
from data.episode_dataset import EpiDataset, CategoriesSampler, DCategoriesSampler
from data.test_dataset import TestDataset
import pandas as pd
from data.transforms import data_transform

from models.utils.comm import get_world_size


class ImgDatasetParam(object):
    ## update dir 'imgroot', "dataroot"
    DATASETS = {
        "imgroot": datasets',
        "dataroot": '/data/xlsa19',
        "image_embedding": 'images_label',
        "class_embedding": 'att'
    }

    @staticmethod
    def get(dataset):
        attrs = ImgDatasetParam.DATASETS
        attrs["imgroot"] = join(attrs["imgroot"], dataset)
        args = dict(
            dataset=dataset
        )
        args.update(attrs)
        return args


def build_dataloader(cfg, is_distributed=False):
    args = ImgDatasetParam.get(cfg.DATASETS.NAME)
    imgroot = args['imgroot']
    dataroot = args['dataroot']
    image_embedding = args['image_embedding']
    dataset = args['dataset']

    matcontent = io.loadmat(dataroot + "/" + dataset + "/" + image_embedding + ".mat")

    img_files = np.squeeze(matcontent['image_files'])
    new_img_files = []
    for img_file in img_files:
        img_path = img_file[0]
        img_path = join(imgroot + "/" + "images/", img_path)
        new_img_files.append(img_path)

    new_img_files = np.array(new_img_files)
    label = matcontent['labels'].astype(int).squeeze() - 1

    # 加载split文件
    if dataset == 'GroupA':
        matcontent = io.loadmat(dataroot + "/split/splits_group_A.mat")
        trainvalloc = matcontent['trainval_loc'].squeeze()
        test_seen_loc = matcontent['test_seen_loc'].squeeze()
        test_unseen_loc = matcontent['test_unseen_loc'].squeeze()
    elif dataset == 'GroupB':
        matcontent = io.loadmat(dataroot + "/split/splits_group_B.mat")
        trainvalloc = matcontent['trainval_loc'].squeeze()
        test_seen_loc = matcontent['test_seen_loc'].squeeze()
        test_unseen_loc = matcontent['test_unseen_loc'].squeeze()
    else:
        matcontent = io.loadmat(dataroot + "/split/splits_group_C.mat")
        trainvalloc = matcontent['trainval_loc'].squeeze()
        test_seen_loc = matcontent['test_seen_loc'].squeeze()
        test_unseen_loc = matcontent['test_unseen_loc'].squeeze()

    # 所以需要将所有索引减1
    trainvalloc = trainvalloc - 1
    test_seen_loc = test_seen_loc - 1
    test_unseen_loc = test_unseen_loc - 1
    # 验证索引有效性
    max_valid_idx = len(new_img_files) - 1
    if (trainvalloc.max() > max_valid_idx or
            test_seen_loc.max() > max_valid_idx or
            test_unseen_loc.max() > max_valid_idx):
        raise ValueError(f"索引超出范围！最大有效索引是 {max_valid_idx}")

    if (trainvalloc.min() < 0 or
            test_seen_loc.min() < 0 or
            test_unseen_loc.min() < 0):
        raise ValueError(f"索引为负数！请检查数据")

    print(f"✓ 索引有效性验证通过")
    
    att_name = 'att'
    matcontent2 = io.loadmat(dataroot + "/allclasses_names.mat")
    cls_name = matcontent2['allclasses_names']

       matcontent3 = io.loadmat(dataroot + "/att/att_value-train.mat")
    matcontent3[att_name] = matcontent3[att_name]
    attribute = matcontent3[att_name][label]

    # 使用转换后的索引提取数据
    train_img = new_img_files[trainvalloc]
    train_label = label[trainvalloc].astype(int)
    train_att = attribute[train_label]

    train_id, idx = np.unique(train_label, return_inverse=True)
    train_att_unique = matcontent3[att_name][train_id]
    train_clsname = cls_name[train_id]

    num_train = len(train_id)
    train_label = idx
    train_id = np.unique(train_label)

    test_img_unseen = new_img_files[test_unseen_loc]
    test_label_unseen = label[test_unseen_loc].astype(int)
    test_id, idx = np.unique(test_label_unseen, return_inverse=True)
    att_unseen = matcontent3[att_name][test_id]
    test_clsname = cls_name[test_id]
    test_label_unseen = idx + num_train
    test_id = np.unique(test_label_unseen)

    # ========== 严格验证数据分割 ==========
    train_label_original = label[trainvalloc].astype(int)
    test_label_seen_original = label[test_seen_loc].astype(int)
    test_label_unseen_original = label[test_unseen_loc].astype(int)

    print(f"\n=== 数据分割统计 ===")
    print(f"训练集样本数: {len(train_label_original)}")
    print(f"训练集唯一标签数: {len(np.unique(train_label_original))}")
    print(f"训练集类别: {sorted(np.unique(train_label_original).tolist())}")

    print(f"\n测试集可见样本数: {len(test_label_seen_original)}")
    print(f"测试集可见唯一标签数: {len(np.unique(test_label_seen_original))}")
    print(f"测试集可见类别: {sorted(np.unique(test_label_seen_original).tolist())}")

    print(f"\n测试集不可见样本数: {len(test_label_unseen_original)}")
    print(f"测试集不可见唯一标签数: {len(np.unique(test_label_unseen_original))}")
    print(f"测试集不可见类别: {sorted(np.unique(test_label_unseen_original).tolist())}")

    # ========== 关键验证：检查重叠 ==========
    train_classes = set(train_label_original)
    test_seen_classes = set(test_label_seen_original)
    test_unseen_classes = set(test_label_unseen_original)

    overlap_train_unseen = train_classes & test_unseen_classes
    overlap_seen_unseen = test_seen_classes & test_unseen_classes

    print(f"\n=== 重叠检查 ===")
    if len(overlap_train_unseen) > 0:
        print(f"错误！训练集和测试不可见的重叠类别数: {len(overlap_train_unseen)}")
        print(f"   重叠类别: {sorted(list(overlap_train_unseen))}")
        print(f"\n请检查：")
        print(f"1. splits_group_B.mat 文件是否是用最新代码生成的？")
        print(f"2. unseen_ids 配置是否为 [5, 9, 21, 31, 37, 40, 47, 53, 80, 86, 96, 100]？")
        raise RuntimeError("数据分割错误：训练集包含unseen类别！")
    else:
        print(f"✓ 训练集和测试不可见集无重叠")

    if len(overlap_seen_unseen) > 0:
        print(f" 错误！测试可见和测试不可见的重叠类别数: {len(overlap_seen_unseen)}")
        print(f"   重叠类别: {sorted(list(overlap_seen_unseen))}")
        raise RuntimeError("数据分割错误：测试可见集和测试不可见集有重叠！")
    else:
        print(f"✓ 测试可见集和测试不可见集无重叠")

    # 这个重叠是正常的（训练集和测试可见集共享类别）
    overlap_train_seen = train_classes & test_seen_classes
    print(f"✓ 训练集和测试可见的共享类别数: {len(overlap_train_seen)} （正常）")
    # ==========================================

    train_test_att = np.concatenate((train_att_unique, att_unseen))
    train_test_id = np.concatenate((train_id, test_id))

    total_classes = len(np.unique(train_test_id))
    print(f"\n总类别数: {total_classes}")
    expected_total = len(train_classes) + len(test_unseen_classes)
    if total_classes != expected_total:
        print(
            f"⚠ 警告：总类别数({total_classes}) ≠ seen类别({len(train_classes)}) + unseen类别({len(test_unseen_classes)}) = {expected_total}")

    test_img_seen = new_img_files[test_seen_loc]
    test_label_seen = label[test_seen_loc].astype(int)
    _, idx = np.unique(test_label_seen, return_inverse=True)
    test_label_seen = idx

    att_unseen = torch.from_numpy(att_unseen).float()
    test_label_seen = torch.tensor(test_label_seen)
    test_label_unseen = torch.tensor(test_label_unseen)
    train_label = torch.tensor(train_label)
    att_seen = torch.from_numpy(train_att_unique).float()
    trian_class_counts = torch.bincount(train_label).float()
    seenclasses = torch.from_numpy(np.unique(train_label.cpu().numpy()))
    unseenclasses = torch.from_numpy(np.unique(test_label_unseen.cpu().numpy()))

    print(f"\n=== 最终统计 ===")
    print(f"训练集类别数: {len(seenclasses)}")
    print(f"测试可见类别数: {len(np.unique(test_label_seen))}")
    print(f"测试不可见类别数: {len(unseenclasses)}")
    print("=" * 60)

    res = {
        'train_label': train_label,
        'train_att': train_att,
        'test_label_seen': test_label_seen,
        'test_label_unseen': test_label_unseen,
        'att_unseen': att_unseen,
        'att_seen': att_seen,
        'train_id': train_id,
        'test_id': test_id,
        'train_test_id': train_test_id,
        'train_clsname': train_clsname,
        'test_clsname': test_clsname,
        'seenclasses': seenclasses,
        'unseenclasses': unseenclasses,
        'trian_class_counts': trian_class_counts
    }

    if cfg.VISUAL:
        attr_path = './attribute/IP102/new_des.csv'
        attr_name = pd.read_csv(attr_path)['new_des']
        res['attr_name'] = attr_name
        res['train_img'] = train_img
        res['test_img_seen'] = test_img_seen
        res['test_img_unseen'] = test_img_unseen

    # train dataloader
    ways = cfg.DATASETS.WAYS
    shots = cfg.DATASETS.SHOTS
    data_aug_train = cfg.SOLVER.DATA_AUG
    img_size = cfg.DATASETS.IMAGE_SIZE
    transforms = data_transform(data_aug_train, size=img_size)

    if cfg.DATALOADER.MODE == 'random':
        dataset = RandDataset(train_img, train_att, train_label, transforms)

        if not is_distributed:
            sampler = torch.utils.data.sampler.RandomSampler(dataset)
            batch = ways * shots
            batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, batch_size=batch, drop_last=True)
            tr_dataloader = torch.utils.data.DataLoader(
                dataset=dataset,
                num_workers=8,
                batch_sampler=batch_sampler,
            )
        else:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
            batch = ways * shots
            tr_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch, sampler=sampler, num_workers=8)

    elif cfg.DATALOADER.MODE == 'episode':
        n_batch = cfg.DATALOADER.N_BATCH
        ep_per_batch = cfg.DATALOADER.EP_PER_BATCH
        dataset = EpiDataset(train_img, train_att, train_label, transforms)
        if not is_distributed:
            sampler = CategoriesSampler(
                train_label,
                n_batch,
                ways,
                shots,
                ep_per_batch
            )
        else:
            sampler = DCategoriesSampler(
                train_label,
                n_batch,
                ways,
                shots,
                ep_per_batch
            )
        tr_dataloader = DataLoader(dataset=dataset, batch_sampler=sampler, num_workers=8, pin_memory=True)

    data_aug_test = cfg.TEST.DATA_AUG
    transforms = data_transform(data_aug_test, size=img_size)
    test_batch_size = cfg.TEST.IMS_PER_BATCH

    if not is_distributed:
        tu_data = TestDataset(test_img_unseen, test_label_unseen, transforms)
        tu_loader = torch.utils.data.DataLoader(
            tu_data, batch_size=test_batch_size, shuffle=False,
            num_workers=4, pin_memory=False)

        ts_data = TestDataset(test_img_seen, test_label_seen, transforms)
        ts_loader = torch.utils.data.DataLoader(
            ts_data, batch_size=test_batch_size, shuffle=False,
            num_workers=4, pin_memory=False)
    else:
        tu_data = TestDataset(test_img_unseen, test_label_unseen, transforms)
        tu_sampler = torch.utils.data.distributed.DistributedSampler(dataset=tu_data, shuffle=False)
        tu_loader = torch.utils.data.DataLoader(
            tu_data, batch_size=test_batch_size, sampler=tu_sampler,
            num_workers=4, pin_memory=False)

        ts_data = TestDataset(test_img_seen, test_label_seen, transforms)
        ts_sampler = torch.utils.data.distributed.DistributedSampler(dataset=ts_data, shuffle=False)
        ts_loader = torch.utils.data.DataLoader(
            ts_data, batch_size=test_batch_size, sampler=ts_sampler,
            num_workers=4, pin_memory=False)

    return tr_dataloader, tu_loader, ts_loader, res