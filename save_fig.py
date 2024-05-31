import torchio as tio
import matplotlib.pyplot as plt
import yaml
from utils import KFoldNNUNetSegmentationDataModule

if __name__ == '__main__':
    with open('configs/evals/kfold_segmentation.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    dm = KFoldNNUNetSegmentationDataModule(config)
    dm.setup('fit')

    cnt = 0

    for data in dm.train_dataloader():
        print(data['image'].shape, data['label'].shape)
        subject = tio.Subject(
            **{f'image_{batch_index}': tio.data.ScalarImage(tensor=data['image'][batch_index]) for batch_index in range(data['image'].shape[0])},
            **{f'label_{batch_index}': tio.data.LabelMap(tensor=data['label'][batch_index].unsqueeze(0)) for batch_index in range(data['label'].shape[0])},
        )
        # save the subject.plot
        subject.plot(figsize=(100, 10))
        fig = plt.gcf()
        fig.savefig(f'subject_iter_{cnt:03d}.png')
        cnt += 1
        #clear the figure
        plt.close('all')
