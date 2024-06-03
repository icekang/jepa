# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch

import torch
import torchio as tio
from torch.utils.data import DataLoader
from pathlib import Path
import json
from typing import List, Tuple
from functools import partial

class KFoldNNUNetSegmentationDataModule(torch.nn.Module):
    def __init__(self,
                    config: dict) -> None:
        self.config = config
        self.fold = self.config['data']['fold']
        self.dataDir = self.config['data']['data_directory'] # /storage_bizon/naravich/nnUNet_Datasets/nnUNet_raw/Dataset301_Calcium_OCT
        if isinstance(self.dataDir, str):
            self.dataDir = Path(self.dataDir)

        self.num_workers = self.config['data']['num_workers']
        self.batch_size = self.config['data']['batch_size']

    def setup(self, stage: str) -> None:
        """Define the split and data before putting them into dataloader

        Args:
            stage (str): Torch Lightning stage ('fit', 'validate', 'predict', ...), not used in the LightningDataModule
        """
        self.preprocess = self.getPreprocessTransform()
        self.augment = self.getAugmentationTransform()
        self.transform = tio.Compose([self.preprocess, self.augment])

        if stage == 'fit' or stage is None:
            trainImages, trainLabels = self._getImagesAndLabels('train')
            valImages, valLabels = self._getImagesAndLabels('val')

            trainSubjects = self._filesToSubject(trainImages, trainLabels)
            valSubjects = self._filesToSubject(valImages, valLabels)

            self.trainSet = tio.SubjectsDataset(trainSubjects, transform=self.transform)
            # trainSampler = tio.data.UniformSampler(
            #     patch_size=self.config['data']['patch_size'],
            # )
            trainSampler = tio.data.LabelSampler(
                patch_size=self.config['data']['patch_size'],
                label_name='label',
                label_probabilities={0: 1, 1: 15},
            )
            self.patchesTrainSet = tio.Queue(
                subjects_dataset=self.trainSet,
                max_length=self.config['data']['queue_max_length'],
                samples_per_volume=self.config['data']['samples_per_volume'],
                sampler=trainSampler,
                num_workers=self.num_workers,
                shuffle_subjects=True,
                shuffle_patches=True,)
            print('=====================================================================================================================\n')
            print('self.patchesTrainSet.iterations_per_epoch // self.batch_size', self.patchesTrainSet.iterations_per_epoch // self.batch_size)
            print('\n=====================================================================================================================')

            if len(valSubjects) == 0:
                valSubjects = trainSubjects
                print("Warning: Validation set is empty, using training set for validation")
            self.valSet = tio.SubjectsDataset(valSubjects, transform=self.preprocess)
            valSampler = tio.data.UniformSampler(
                patch_size=self.config['data']['patch_size'],
            )
            self.patchesValSet = tio.Queue(
                subjects_dataset=self.valSet,
                max_length=self.config['data']['queue_max_length'],
                samples_per_volume=self.config['data']['samples_per_volume'],
                sampler=valSampler,
                num_workers=self.num_workers,
                shuffle_subjects=False,
                shuffle_patches=False,
            )

        if stage == 'test':
            testImages, testLabels = self._getImagesAndLabels('test')

            testSubjects = self._filesToSubject(testImages, testLabels)
            self.testSubjectGridSamplers = [tio.inference.GridSampler(
                subject=testSubject,
                patch_size=self.config['data']['patch_size'],
                patch_overlap=(s//2 for s in self.config['data']['patch_size'])) for testSubject in testSubjects]
            self.testAggregators = [tio.inference.GridAggregator(gridSampler) for gridSampler in self.testSubjectGridSamplers]

    @staticmethod
    def Gray2RGB(tensor5D: torch.Tensor):
        B, C, H, W, T = tensor5D.shape

        assert C == 1, f"Input tensor must have 1 channel, got {tensor5D.shape[1]}"
        return torch.repeat_interleave(tensor5D, repeats=3, dim=1)

    @staticmethod
    def BCHWT2BCTHW(tensor5D: torch.Tensor):
        B, C, H, W, T = tensor5D.shape
        assert tensor5D.dim() == 5, f"Input tensor must be 5D, got {tensor5D.dim()}"
        return tensor5D.permute(0, 1, 4, 2, 3)

    @staticmethod
    def collate_fn(batch, test=False):
        compose4VJEPA = lambda x: KFoldNNUNetSegmentationDataModule.BCHWT2BCTHW(KFoldNNUNetSegmentationDataModule.Gray2RGB(x))
        collated_batch = {
            'image': torch.stack([data['image'][tio.DATA] for data in batch], dim=0),
            # 'image_from_labels': torch.stack([data['image_from_labels'][tio.DATA] for data in batch], dim=0),
            'label': torch.stack([data['label'][tio.DATA] for data in batch], dim=0),
        }
        collated_batch['image'] = compose4VJEPA(collated_batch['image'])
        collated_batch['label'] = KFoldNNUNetSegmentationDataModule.BCHWT2BCTHW(collated_batch['label']).squeeze(1)
        if test:
            collated_batch['location'] = torch.stack([data[tio.LOCATION] for data in batch], dim=0)

        return collated_batch

    def train_dataloader(self):
        return DataLoader(self.patchesTrainSet, batch_size=self.batch_size, num_workers=0, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.patchesValSet, batch_size=self.batch_size, num_workers=0, collate_fn=self.collate_fn)

    def test_dataloader(self) -> Tuple[List[DataLoader], List[tio.GridSampler]]:
        return [DataLoader(testSubjectGridSampler, batch_size=self.batch_size, num_workers=0, collate_fn=partial(self.collate_fn, test=True)) for testSubjectGridSampler in self.testSubjectGridSamplers], self.testSubjectGridSamplers

    def getPreprocessTransform(self):
        H, W, D = self.config['data']['patch_size']
        preprocess = tio.Compose([
            tio.CropOrPad((500, 500, 400)),
            tio.transforms.Resize(target_shape=[H, W, 400]),
            tio.RescaleIntensity((0, 1)),
        ])
        return preprocess

    def getAugmentationTransform(self):
        H, W, D = self.config['data']['patch_size']
        augment = tio.Compose([
            tio.Pad((H // 2, W // 2, D // 2)),
            tio.RandomFlip(axes=(0, 1, 2), flip_probability=0.2),
            tio.RandomAffine(scales=(0.7, 1.4), degrees=0, isotropic=True, translation=0, center='image', default_pad_value=0, p=0.2, image_interpolation='bspline'),
            tio.RandomAffine(scales=0, degrees=(0, 0, 180) , isotropic=True, translation=0, center='image', default_pad_value=0, p=0.2, image_interpolation='bspline'),
            tio.RandomAffine(scales=0, degrees=0 , isotropic=True, translation=60, center='image', default_pad_value=0, p=0.2, image_interpolation='bspline'),
            tio.transforms.RandomAnisotropy(axes=(0, 1, 2), downsampling=(1, 1.5), scalars_only=True, p=0.2, image_interpolation='bspline'),
            tio.transforms.RandomBlur(std=(0.5, 1.), p=0.2), # Like nnUNet
            tio.RandomNoise(mean=0, std=(0, 0.1), p=0.1), # Like nnUNet
            tio.transforms.RandomGamma(log_gamma=(0.7, 1.5), p=0.1),
        ])
        return augment

    def _filesToSubject(self, imageFiles: List[Path], labelFiles: List[Path]) -> List[tio.Subject]:
        """Convert image and label files to TorchIO subjects

        Args:
            imageFiles (List[Path]): List of image files
            labelFiles (List[Path]): List of label files

        Returns:
            List[tio.Subject]: List of TorchIO subjects
        """
        subjects = []
        for imageFile, labelFile in zip(imageFiles, labelFiles):
            subject = tio.Subject(
                image=tio.ScalarImage(str(imageFile)),
                label=tio.LabelMap(str(labelFile)),
                name=imageFile.stem.split('_')[0]
            )
            subjects.append(subject)
        return subjects

    def _getSplit(self) -> Tuple[List[str], List[str]]:
        """Get the train and validation split for the current fold and split

        Returns:
            Tuple[List[str], List[str]]: List of train and validation unique case IDs
        """
        dataSetName = self.dataDir.stem
        if 'scale_path' in self.config['data']:
            splitPath = Path(self.config['data']['scale_path'])
        else:
            splitPath = self.dataDir / '..' / '..' / 'nnUNet_preprocessed' / dataSetName / 'splits_final.json'
        assert splitPath.exists(), f"Split file {splitPath} does not exist"
        with open(splitPath, 'r') as f:
            splits = json.load(f)
            train = splits[self.fold]['train']
            val = splits[self.fold]['val']

        return train, val

    def _getImagesAndLabels(self, split: str) -> Tuple[List[Path], List[Path]]:
        """Get the image and label files for the current fold and split

        Args:
            split (str): 'train', 'val', or 'test'

        Returns:
            Tuple[List[Path], List[Path]]: List of image and label path files
        """
        train, val = self._getSplit()
        train.sort()
        val.sort()

        if split in {'train', 'val'}:
            imageDir = self.dataDir / 'imagesTr'
            labelDir = self.dataDir / 'labelsTr'
        else:
            imageDir = self.dataDir / 'imagesTs'
            labelDir = self.dataDir / 'labelsTs'
        
        assert imageDir.exists(), f"Image directory {imageDir} does not exist"
        assert labelDir.exists(), f"Label directory {labelDir} does not exist"

        imageFiles = sorted(list(imageDir.glob('*.nii.gz')))
        labelFiles = sorted(list(labelDir.glob('*.nii.gz')))

        assert len(imageFiles) == len(labelFiles), f"Number of images and labels do not match: {len(imageFiles)} != {len(labelFiles)}"

        if split == 'train':
            caseFilter = train
        elif split == 'val':
            caseFilter = val
        else:
            caseFilter = None
        if caseFilter is not None:
            imageFiles = [f for f in imageFiles if f.stem.split('_')[0] in caseFilter]
            labelFiles = [f for f in labelFiles if f.stem.replace('.nii', '') in caseFilter]

        assert len(imageFiles) == len(labelFiles), f"Number of images and labels AFTER filtering do not match: {len(imageFiles)} != {len(labelFiles)}, filter={caseFilter}"

        return imageFiles, labelFiles