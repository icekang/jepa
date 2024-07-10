import torch
import torchio as tio
from torch.utils.data import DataLoader

from pathlib import Path
import os
import json
from typing import List, Tuple
from functools import partial
import pandas as pd

class KFoldNNUNetTabularDataModule(torch.nn.Module):
    def __init__(self,
                 config: dict) -> None:
        self.config = config
        self.fold = self.config['data']['fold']
        self.dataDir = self.config['data']['data_directory'] # /storage_bizon/naravich/Unlabeled_OCT_by_CADx/NiFTI/
        self.tabularDataDir = Path(self.config['data']['tabular_data_directory'])
        
        self.inputModality = self.config['data']['input_modality'] # ('pre', 'post', 'final')
        self.outputModality = self.config['data']['output_modality'] # ('pre', 'post', 'final')
        self.outputMetrics: List[str] = self.config['data']['output_metrics']
        # Just for now
        self.modalityToDataframePath = {
            'pre': self.tabularDataDir / 'Pre_IVL.csv',
            'post': self.tabularDataDir / 'Post_IVL.csv',
            'final': self.tabularDataDir / 'Post_Stent.csv'
        }
        self.modalityToName = {
            'pre': 'Pre_IVL',
            'post': 'Post_IVL',
            'final': 'Post_Stent'
        }

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

        # create subject from the CSV!
        outputModalityDf = pd.read_csv(self.modalityToDataframePath[self.outputModality])
        inputName = self.modalityToName[self.inputModality]
        outputModalityDf = outputModalityDf[['USUBJID'] + self.outputMetrics + [f'{inputName}_image_path']]

        if self.config['data']['nan_handling'] == 'drop':
            outputModalityDf.dropna(inplace=True)
        elif self.config['data']['nan_handling'] == 'mean':
            outputModalityDf.fillna(outputModalityDf.mean(), inplace=True)
        elif self.config['data']['nan_handling'] == 'median':
            outputModalityDf.fillna(outputModalityDf.median(), inplace=True)
        elif self.config['data']['nan_handling'] == 'zero':
            outputModalityDf.fillna(0, inplace=True)

        outputModalityDf[f'{inputName}_image_path'] = outputModalityDf[f'{inputName}_image_path'].apply(lambda x: f'{x}.nii.gz')

        if self.config['data']['target_normalization'] == 'minmax':
            # Get the train split to normalize the target
            train_subject_ids, _ = self._getSplit('fit', outputModalityDf=outputModalityDf)
            trainDF = outputModalityDf[outputModalityDf['USUBJID'].isin(train_subject_ids)]
            outputModalityDf[self.outputMetrics] = (outputModalityDf[self.outputMetrics] - trainDF[self.outputMetrics].min(axis=0)) / (trainDF[self.outputMetrics].max(axis=0) - trainDF[self.outputMetrics].min(axis=0))
        
        train_subject_ids, val_subject_ids = self._getSplit(stage, outputModalityDf=outputModalityDf)

        if stage == 'test':
            test_subject_ids = train_subject_ids

        # Create the subjects
        if stage == 'fit':
            if self.config['data']['overfit']:
                val_subject_ids = train_subject_ids.copy()
                train_subject_ids = train_subject_ids[:1]
                val_subject_ids = val_subject_ids[:1]
            trainSubjects = []
            for _, row in outputModalityDf[outputModalityDf['USUBJID'].isin(train_subject_ids)].iterrows():
                assert (self.dataDir / row[f'{inputName}_image_path']).exists(), '{} does not exist'.format(self.dataDir / row[f'{inputName}_image_path'])
                subject = tio.Subject(
                    **{metric: row[metric] for metric in self.outputMetrics},
                    image=tio.ScalarImage(self.dataDir / row[f'{inputName}_image_path'])
                )
                trainSubjects.append(subject)
            
            valSubjects = []
            for _, row in outputModalityDf[outputModalityDf['USUBJID'].isin(val_subject_ids)].iterrows():
                assert (self.dataDir / row[f'{inputName}_image_path']).exists(), '{} does not exist'.format(self.dataDir / row[f'{inputName}_image_path'])
                subject = tio.Subject(
                    **{metric: row[metric] for metric in self.outputMetrics},
                    image=tio.ScalarImage(self.dataDir / row[f'{inputName}_image_path'])
                )
                valSubjects.append(subject)

            self.trainSet = tio.SubjectsDataset(subjects=trainSubjects, transform=self.transform)
            self.valSet = tio.SubjectsDataset(subjects=valSubjects, transform=self.preprocess)
            self.patchesTrainSet = tio.Queue(
                subjects_dataset=self.trainSet,
                max_length=self.config['data']['queue_max_length'],
                samples_per_volume=self.config['data']['samples_per_volume'],
                sampler=tio.UniformSampler(patch_size=self.config['data']['patch_size']),
                num_workers=self.num_workers,
                shuffle_subjects=True,
                shuffle_patches=True,)
            self.patchesValSet = tio.Queue(
                subjects_dataset=self.valSet,
                max_length=len(valSubjects) if not self.config['data']['overfit'] else self.config['data']['queue_max_length'],
                samples_per_volume=1 if not self.config['data']['overfit'] else self.config['data']['samples_per_volume'],
                sampler=tio.UniformSampler(patch_size=self.config['data']['patch_size']),
                num_workers=self.num_workers,
                shuffle_subjects=False,
                shuffle_patches=False,)
        elif stage == 'test':
            testSubjects = []
            for _, row in outputModalityDf[outputModalityDf['USUBJID'].isin(test_subject_ids)].iterrows():
                subject = tio.Subject(
                    **{metric: row[metric] for metric in self.outputMetrics},
                    image=tio.ScalarImage(self.dataDir / row[f'{inputName}_image_path'])
                )
                testSubjects.append(subject)
            self.testSet = tio.SubjectsDataset(subjects=testSubjects, transform=self.preprocess)
            sampler = tio.UniformSampler(patch_size=self.config['data']['patch_size'])
            self.patchesTestSet = tio.Queue(
                subjects_dataset=self.testSet,
                max_length=len(testSubjects),
                samples_per_volume=1,
                sampler=sampler,
                num_workers=self.num_workers,
                shuffle_subjects=False,
                shuffle_patches=False,)

    @staticmethod    
    def SampleEveryOtherFrame(tensor4D: torch.Tensor):
        return tensor4D[:, :, :, ::3]

    def getPreprocessTransform(self):
            preprocess = tio.Compose([
                tio.transforms.CropOrPad(target_shape=(500, 500, 375)),
                tio.transforms.Lambda(KFoldNNUNetTabularDataModule.SampleEveryOtherFrame),
                tio.transforms.Resize(target_shape=self.config['data']['patch_size'], image_interpolation='bspline'),
                tio.transforms.RescaleIntensity(out_min_max=(0, 1))
            ])
            return preprocess

    def getAugmentationTransform(self):
        if self.config['data']['overfit']:
            return tio.Compose([])
        augment = tio.Compose([
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
    

    def _getSplit(self, stage: str, outputModalityDf: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Get the train and validation split for the current fold and split

        Returns:
            Tuple[List[str], List[str]]: List of train and validation unique case IDs, if stage is 'test' return test case IDs, and None
        """
        if not os.path.exists(self.tabularDataDir / 'splits_final.json'):
            from sklearn.model_selection import train_test_split
            subject_ids = outputModalityDf['USUBJID'].tolist()
            train_subject_ids, test_subject_ids = train_test_split(subject_ids, test_size=0.2, random_state=0)
            with open(self.tabularDataDir / 'test.json', 'w') as f:
                json.dump(test_subject_ids, f, indent=4)
            splits = []
            for fold in range(3):
                fold_train_subject_ids, fold_val_subject_ids = train_test_split(train_subject_ids, test_size=0.2, random_state=fold)
                splits.append({
                    'train': fold_train_subject_ids,
                    'val': fold_val_subject_ids,
                })
            with open(self.tabularDataDir / 'splits_final.json', 'w') as f:
                json.dump(splits, f, indent=4)

        if stage == 'fit':
            with open(self.tabularDataDir / 'splits_final.json', 'r') as f:
                splits = json.load(f)
            train_subject_ids = splits[self.fold]['train']
            val_subject_ids = splits[self.fold]['val']
            return train_subject_ids, val_subject_ids

        elif stage == 'test':
            with open(self.tabularDataDir / 'test.json', 'r') as f:
                test_subject_ids = json.load(f)
            return test_subject_ids, None

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
        compose4VJEPA = lambda x: KFoldNNUNetTabularDataModule.BCHWT2BCTHW(KFoldNNUNetTabularDataModule.Gray2RGB(x))
        collated_batch = {
            metric: torch.tensor([[data[metric]] for data in batch]) for metric in batch[0].keys() if metric != 'image' and metric != 'location' # labels
        }
        collated_batch['image'] = torch.stack([data['image'][tio.DATA] for data in batch], dim=0)
        collated_batch['image'] = compose4VJEPA(collated_batch['image'])

        if test:
            collated_batch['location'] = torch.stack([data[tio.LOCATION] for data in batch], dim=0)

        return collated_batch

    def train_dataloader(self):
        return DataLoader(self.patchesTrainSet, batch_size=self.batch_size, num_workers=0, collate_fn=self.collate_fn)
    
    def val_dataloader(self):
        return DataLoader(self.patchesValSet, batch_size=self.batch_size, num_workers=0, collate_fn=self.collate_fn)

    def test_dataloader(self) -> Tuple[List[DataLoader], List[tio.GridSampler]]:
        return DataLoader(self.patchesTestSet, batch_size=self.batch_size, num_workers=0, collate_fn=self.collate_fn)