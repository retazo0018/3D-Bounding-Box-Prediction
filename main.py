'''
    Copyright (c) 2025 Ashwin Murali <ashwin.cse18@gmail.com>
    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
'''

import argparse
import torch
import math
from data_viz import plot_losses
from torchinfo import summary
from torch.utils.data import DataLoader
from data_prepare import load_and_prepare_data, Custom3DBBoxDataset
from model import MultiObject3DBBoxModel, hybrid_3d_bbox_loss
from rich.console import Console
from rich.panel import Panel
from transformers import get_cosine_schedule_with_warmup
console = Console()


def main(data_dir, epochs, batch_size):
    """
    Main training entry point.

    Args:
        data_dir (str): Path to the dataset directory.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size used for training.

    This function prepares the data, starts the training process and saves logs, results and model.
    """
    MAX_INSTANCES = 25
    FIXED_DIMENSION = (640, 640)

    console.print(Panel.fit("🚀 [bold green]3D Bounding Box Prediction[/bold green]", border_style="green"))

    console.print(f"[bold yellow]Starting Data Preprocessing...[/bold yellow]")
    dataset = Custom3DBBoxDataset(load_and_prepare_data(data_dir, MAX_INSTANCES, FIXED_DIMENSION))
    console.print(f"[bold green]Data preprocessing complete! Loaded {len(dataset)} samples. [/bold green]")
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    dl_model = MultiObject3DBBoxModel(MAX_INSTANCES)
    console.print(f"[bold green]Model Summary: [/bold green]")
    summary(dl_model, input_data=(torch.zeros([1, 3, 512, 512]), torch.zeros([1, 25, 512, 512]), torch.zeros([1, 3, 512, 512])))
    
    optimizer = torch.optim.AdamW(dl_model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=5,
        num_training_steps=math.ceil(len(train_dataset) / batch_size)  # e.g., epochs * steps_per_epoch
    )
    dl_model.train()
    console.print(f"[bold yellow]Starting Model Training...[/bold yellow]")

    loss_per_epoch = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        for i, batch in enumerate(train_dataloader):
            rgb, mask, pc, bbox3d = batch
            optimizer.zero_grad()
            pred_boxes = dl_model(rgb, mask, pc)
            loss = hybrid_3d_bbox_loss(pred_boxes, bbox3d)
            console.print(
            f"🔄 [bold cyan]Training Epoch {epoch+1} [/bold cyan] "
            f"┃ Batch: [bold yellow]{i+1}[/bold yellow] ended with "
            f"[bold red]{loss:.4f}[/bold red] loss.")
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss+=loss

        console.print(f"[bold green]Epoch: {epoch+1} ended with loss {epoch_loss/len(train_dataloader)}. [/bold green]")
        loss_per_epoch.append(epoch_loss.detach()/len(train_dataloader))

    console.print(f"[bold green]Model Training complete! [/bold green]")
    console.print(f"[bold yellow]🔄 Starting Evaluation ... [/bold yellow]")
    plot_losses(loss_per_epoch)

    with torch.no_grad():
        test_loss = 0
        dl_model.eval()
        for i, batch in enumerate(test_dataloader):
            rgb, mask, pc, bbox3d = batch
            pred_boxes = dl_model(rgb, mask, pc)
            loss = hybrid_3d_bbox_loss(pred_boxes, bbox3d)
            test_loss+=loss

        console.print(f"[bold green]Model Evaluation Complete. Loss obtained: {test_loss/len(test_dataloader)}. [/bold green]")

    torch.save(dl_model, 'model.pt')
    console.print(f"[bold green]Model Saved Successfully! [/bold green]")


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Training script for 3D bounding box pre")
    parser.add_argument("--data_dir", type=str, default="./data", help="Path to the data directory")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    
    args = parser.parse_args()
    main(data_dir=args.data_dir, epochs=args.epochs, batch_size=args.batch_size)    
