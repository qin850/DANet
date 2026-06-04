import os
import random
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import scipy.io as sio


class DatasetSplitter:
    """
    生成 splits_group_X.mat 和 images_label.mat 的工具。
    严格保证：
      - trainval 和 test_seen 仅包含 seen 类图片（不含 any unseen 类）
      - test_unseen 仅包含 unseen 类图片
      - 三集合互斥
      - 保存的索引为 1-based（MATLAB 风格）
    """

    def __init__(self, image_dir, label_file, seed=42):
        self.image_dir = Path(image_dir)
        self.label_file = Path(label_file)
        self.seed = int(seed)
        random.seed(self.seed)
        np.random.seed(self.seed)

        # key: 文件名含扩展名（保证唯一），value: class_id (int)
        self.image_to_label = {}
        # 全局图片列表（按稳定顺序），以及1-based索引
        self.all_images = []
        self.image_to_index = {}

    def load_data(self):
        """从单个标签文件加载：每行至少包含 图片路径/名 和 类别ID(整型)。"""
        if not self.label_file.exists():
            raise FileNotFoundError(f"标签文件不存在: {self.label_file}")

        with open(self.label_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]

        for i, line in enumerate(lines, 1):
            parts = line.split()
            if len(parts) < 2:
                print(f"跳过第{i}行（格式不够）: {line}")
                continue
            img_path = parts[0]
            img_name = Path(img_path).name  # 使用含扩展名的文件名做 key
            try:
                class_id = int(parts[-1])
            except Exception:
                print(f"第{i}行类ID解析失败，跳过: {line}")
                continue

            # 若文件名重复，自动添加后缀保持唯一
            if img_name in self.image_to_label:
                cnt = 1
                new_name = f"{img_name}_dup{cnt}"
                while new_name in self.image_to_label:
                    cnt += 1
                    new_name = f"{img_name}_dup{cnt}"
                print(f"注意: 文件名重复: '{img_name}' -> 改为 '{new_name}' 保证唯一键")
                img_name = new_name

            self.image_to_label[img_name] = class_id

        if not self.image_to_label:
            raise ValueError("未解析到任何图片标签，请检查标签文件格式。")

        # 稳定排序（保证复现性）并生成 1-based 索引
        self.all_images = sorted(self.image_to_label.keys())
        self.image_to_index = {img: idx + 1 for idx, img in enumerate(self.all_images)}

        print(f"加载完成：{len(self.all_images)} 张图片，{len(set(self.image_to_label.values()))} 个类别。")

    def create_split(self, split_config, output_dir, group_name):
        """
        split_config: dict 包含:
           - unseen_ids: list of int (declared unseen class ids)
           - train_size: int
           - test_seen_size: int
           - test_unseen_size: int
        output_dir: 保存目录
        group_name: 字母 or 名称 (用于生成 group_x 文件夹)
        """
        outdir = Path(output_dir) / f"group_{group_name}"
        outdir.mkdir(parents=True, exist_ok=True)

        unseen_declared = set(int(x) for x in split_config["unseen_ids"])
        train_size = int(split_config["train_size"])
        test_seen_size = int(split_config["test_seen_size"])
        test_unseen_size = int(split_config["test_unseen_size"])

        print(f"\n声明的 unseen 类别: {sorted(list(unseen_declared))}")

        # 按类分配图片池
        images_per_class = defaultdict(list)
        for img, cid in self.image_to_label.items():
            images_per_class[int(cid)].append(img)

        all_class_ids = set(images_per_class.keys())

        # **关键修改：严格区分 seen 和 unseen 类别**
        # unseen 类别：在声明列表中的类别
        unseen_class_ids = sorted(list(all_class_ids & unseen_declared))
        # seen 类别：不在 unseen 声明列表中的所有其他类别
        seen_class_ids = sorted(list(all_class_ids - unseen_declared))

        missing_unseen = sorted(list(unseen_declared - all_class_ids))
        if missing_unseen:
            print(f"警告: 声明为 unseen 的类在数据中未找到: {missing_unseen}")

        print(f"实际 seen 类别数: {len(seen_class_ids)}")
        print(f"实际 unseen 类别数: {len(unseen_class_ids)}")
        print(f"seen 类别: {seen_class_ids}")
        print(f"unseen 类别: {unseen_class_ids}")

        # **严格构造 seen 和 unseen 图片池**
        seen_images = []
        for cid in seen_class_ids:
            seen_images.extend(images_per_class[cid])

        unseen_images = []
        for cid in unseen_class_ids:
            unseen_images.extend(images_per_class[cid])

        print(f"\nGroup {group_name}: seen images={len(seen_images)} in {len(seen_class_ids)} classes")
        print(f"                   unseen images={len(unseen_images)} in {len(unseen_class_ids)} classes")

        # 验证：确保两个池没有交集
        seen_set_verify = set(seen_images)
        unseen_set_verify = set(unseen_images)
        intersection = seen_set_verify & unseen_set_verify
        if intersection:
            print(f"错误：发现 seen 和 unseen 池有交集，共 {len(intersection)} 张图片！")
            raise RuntimeError("seen 和 unseen 图片池有交集，请检查数据！")

        # 随机打乱
        rng = random.Random(self.seed)
        rng.shuffle(seen_images)
        rng.shuffle(unseen_images)

        # **严格从各自池中采样**
        final_train = seen_images[:train_size]
        final_test_seen = seen_images[train_size:train_size + test_seen_size]
        final_test_unseen = unseen_images[:test_unseen_size]

        # **严格验证：确保没有 unseen 类混入 train 或 test_seen**
        def get_classes(img_list):
            return set(self.image_to_label[img] for img in img_list)

        train_classes = get_classes(final_train)
        test_seen_classes = get_classes(final_test_seen)
        test_unseen_classes = get_classes(final_test_unseen)

        print(f"\n最终集合大小:")
        print(f"  train: {len(final_train)} 张图片, {len(train_classes)} 个类别")
        print(f"  test_seen: {len(final_test_seen)} 张图片, {len(test_seen_classes)} 个类别")
        print(f"  test_unseen: {len(final_test_unseen)} 张图片, {len(test_unseen_classes)} 个类别")

        # **严格检查类别归属**
        train_bad = train_classes & unseen_declared
        test_seen_bad = test_seen_classes & unseen_declared
        test_unseen_bad = test_unseen_classes - unseen_declared

        if train_bad or test_seen_bad or test_unseen_bad:
            print("\n错误: 划分不满足要求！")
            if train_bad:
                print(f"  train 包含 unseen 类: {sorted(list(train_bad))}")
            if test_seen_bad:
                print(f"  test_seen 包含 unseen 类: {sorted(list(test_seen_bad))}")
            if test_unseen_bad:
                print(f"  test_unseen 包含非-unseen 类: {sorted(list(test_unseen_bad))}")
            raise RuntimeError("划分校验失败！")

        print("\n✓ 验证通过: 所有集合的类别归属正确")
        print(f"  train 类别: {sorted(list(train_classes))}")
        print(f"  test_seen 类别: {sorted(list(test_seen_classes))}")
        print(f"  test_unseen 类别: {sorted(list(test_unseen_classes))}")

        # 按全局索引排序
        def sort_by_global(img_list):
            return sorted(img_list, key=lambda x: self.image_to_index[x])

        final_train = sort_by_global(final_train)
        final_test_seen = sort_by_global(final_test_seen)
        final_test_unseen = sort_by_global(final_test_unseen)

        # 构建 index arrays (1-based)
        train_loc = np.array([self.image_to_index[img] for img in final_train], dtype=np.int32).reshape(-1, 1)
        test_seen_loc = np.array([self.image_to_index[img] for img in final_test_seen], dtype=np.int32).reshape(-1, 1)
        test_unseen_loc = np.array([self.image_to_index[img] for img in final_test_unseen], dtype=np.int32).reshape(-1,
                                                                                                                    1)

        # labels_array 按全局顺序
        labels_array = np.array([self.image_to_label[img] for img in self.all_images], dtype=np.int32).reshape(-1, 1)

        # image_files
        pad = max(5, len(str(len(self.all_images))))
        image_files_list = [f"{self.image_to_index[img]:0{pad}d}.jpg" for img in self.all_images]
        image_files_array = np.array(image_files_list, dtype=object).reshape(-1, 1)

        # 保存 mat
        splits_path = outdir / f"splits_group_{group_name}.mat"
        sio.savemat(splits_path, {
            'trainval_loc': train_loc,
            'test_seen_loc': test_seen_loc,
            'test_unseen_loc': test_unseen_loc
        })

        images_label_path = outdir / "images_label.mat"
        sio.savemat(images_label_path, {
            'image_files': image_files_array,
            'labels': labels_array
        })

        print(f"\n保存成功:")
        print(f"  {splits_path}")
        print(f"  {images_label_path}")

        # 最终验证
        self._verify_saved(splits_path, images_label_path, unseen_declared)

        return splits_path

    def _verify_saved(self, splits_mat_path, images_label_path, unseen_declared):
        """读取保存的 .mat 并验证"""
        mc = sio.loadmat(str(splits_mat_path))
        ml = sio.loadmat(str(images_label_path))

        train_loc = np.array(mc['trainval_loc']).squeeze()
        test_seen_loc = np.array(mc['test_seen_loc']).squeeze()
        test_unseen_loc = np.array(mc['test_unseen_loc']).squeeze()
        labels = np.squeeze(ml['labels']).astype(int)

        # 转换为 0-based 索引
        train_idx = train_loc - 1
        seen_idx = test_seen_loc - 1
        unseen_idx = test_unseen_loc - 1

        train_cls = set(labels[train_idx].tolist())
        seen_cls = set(labels[seen_idx].tolist())
        unseen_cls = set(labels[unseen_idx].tolist())

        print(f"\n=== 保存文件验证 ===")
        print(f"train 类别数: {len(train_cls)}")
        print(f"test_seen 类别数: {len(seen_cls)}")
        print(f"test_unseen 类别数: {len(unseen_cls)}")

        bad1 = train_cls & unseen_declared
        bad2 = seen_cls & unseen_declared
        bad3 = unseen_cls - unseen_declared

        if bad1 or bad2 or bad3:
            print("\n保存文件验证失败:")
            if bad1: print(f"  train 包含 unseen 类: {sorted(list(bad1))}")
            if bad2: print(f"  test_seen 包含 unseen 类: {sorted(list(bad2))}")
            if bad3: print(f"  test_unseen 包含非-unseen 类: {sorted(list(bad3))}")
            raise RuntimeError("保存的文件未通过验证！")
        else:
            print("保存文件验证通过！")

if __name__ == "__main__":
    IMAGE_DIR = "ip102_v1.1/images"
    LABEL_FILE = "labels.txt"
    OUTPUT_DIR = "ip102_v1.1/split1"

    split_cfg_A = {
        "unseen_ids": [5, 9, 21, 31, 37, 40, 47, 53, 80, 86, 96, 100],
        "train_size": 48115,
        "test_seen_size": 20687,
        "test_unseen_size": 1932
    }
  
    ds = DatasetSplitter(IMAGE_DIR, LABEL_FILE, seed=1)
    ds.load_data()
    ds.create_split(split_cfg_A, OUTPUT_DIR, "A")