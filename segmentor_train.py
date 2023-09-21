import torchvision
from torchview import draw_graph
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import torch.optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torchmetrics.functional import dice
from dataset_preprocessor.preprocessor import Preprocessor
from segmentor.LinkNet import LinkNet
from dataset_preprocessor.utils import defaultdict, display_image_grid


class MetricMonitor:
    def __init__(self, float_precision=3):
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val):
        metric = self.metrics[metric_name]

        metric["val"] += val
        metric["count"] += 1
        metric["avg"] = metric["val"] / metric["count"]

    def __str__(self):
        return " | ".join(
            [
                "{metric_name}: {avg:.{float_precision}f}".format(
                    metric_name=metric_name, avg=metric["avg"], float_precision=self.float_precision
                )
                for (metric_name, metric) in self.metrics.items()])


def create_model(params):
    model = LinkNet(3)
    model = model.to(params["device"])
    return model


def fit(model, train_dataset, val_dataset, params):
    torch.cuda.empty_cache()
    criterion = nn.CrossEntropyLoss().to(params['device'])
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])

    history = pd.DataFrame(columns=['end_loss', 'end_correct', 'end_dice', 'end_val_loss', 'end_val_correct', 'end_val_dice'])

    for epoch in range(1, params["epochs"] + 1):

        loss = 0
        correct = 0
        dice_score = 0
        train_loss = 0
        train_correct = 0
        train_dice_score = 0
        val_loss = 0
        val_correct = 0
        val_dice_score = 0

        #Train
        metric_monitor = MetricMonitor()
        stream = tqdm(train_loader)
        for i, (images, targets) in enumerate(stream, start=1):
            images = images.to(params["device"], non_blocking=True)
            targets = targets.to(params["device"], non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, targets)
            metric_monitor.update("Loss", loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            stream.set_description("Epoch: {epoch}. Train.  {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor))

        train_loss = loss.item()
        _, pred = torch.max(outputs, 1)
        correct = torch.mean((pred == targets).type(torch.float64))
        dice_score = dice(pred, targets, average='macro', num_classes=3)
        print("-> Epoch: {:.1f}. Train.  Dice Score: {:.3f}  Accuracy: {:.3f}".format(epoch, dice_score, correct.cpu().numpy()))


        with torch.no_grad():
            stream = tqdm(val_loader)
            for i, (images, targets) in enumerate(stream, start=1):
                images = images.to(params["device"], non_blocking=True)
                targets = targets.to(params["device"], non_blocking=True)
                outputs = model(images)
                loss = criterion(outputs, targets)
                metric_monitor.update("Loss", loss.item())
                stream.set_description("Epoch: {epoch}. Validation.  {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor))
            val_loss = loss.item()
            _, pred = torch.max(outputs, 1)
            val_correct = torch.mean((pred == targets).type(torch.float32))
            val_dice_score = dice(pred, targets, average='macro', num_classes=3)
            print("-> Epoch: {:.1f}. Validation.  Dice Score: {:.3f}  Accuracy: {:.3f}".format(epoch, val_dice_score, val_correct.cpu().numpy()))


        history.loc[len(history.index)] = [train_loss, correct.cpu().numpy(),
                                           dice_score.cpu().numpy(), val_loss,
                                           val_correct.cpu().numpy(), val_dice_score.cpu().numpy()]

    return history


def predict(model, params, test_dataset, batch_size):
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=params["num_workers"], pin_memory=True,
    )
    model.eval()
    predictions = []
    with torch.no_grad():
        for images, (original_heights, original_widths) in test_loader:
            images = images.to(params["device"], non_blocking=True)
            output = model(images).squeeze()
            _, predicted_masks = torch.max(output, 1)
            predicted_masks = predicted_masks.cpu().numpy()
            for predicted_mask, original_height, original_width in zip(
                    predicted_masks, original_heights.numpy(), original_widths.numpy()
            ):
                predictions.append((predicted_mask, original_height, original_width))
    return predictions


if __name__ == '__main__':
    preprocessor = Preprocessor('/localhome/aaa324/Project/dataset/oxford-iiit-pet')
    data = preprocessor.preprocessor()
    train_dataset, val_dataset, test_dataset = data['train_dataset'], data['val_dataset'], data['test_dataset']
    params = {
        "device": "cuda",
        "lr": 0.001,
        "batch_size": 32,
        "num_workers": 2,
        "epochs": 17, #15
    }

    train_loader = DataLoader(
        train_dataset,
        batch_size=params["batch_size"],
        shuffle=True,
        num_workers=params["num_workers"],
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=params["batch_size"],
        shuffle=True,
        num_workers=params["num_workers"],
        pin_memory=False,
    )

    model = create_model(params)
    history = fit(model, train_loader, val_loader, params)

    model_graph = draw_graph(model, input_size=(1,3,256,256), expand_nested=True)
    predictions = predict(model, params, test_dataset, batch_size=16)
    predicted_masks = []

    for predicted_256x256_mask, original_height, original_width in predictions:
        full_sized_mask = A.resize(
            predicted_256x256_mask, height=original_height, width=original_width, interpolation=cv2.INTER_NEAREST)
        predicted_masks.append(full_sized_mask)

    test_images_filenames = data['test_images_filenames']
    display_image_grid(test_images_filenames[:10], data['images_directory'], data['masks_directory'], predicted_masks=predicted_masks)


