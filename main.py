
import torch
from torch.utils.data import DataLoader
from data_viz import plot_losses
from data_prepare import prepare_data, Custom3DBBoxDataset
from model import MultiObject3DBBoxModel, hybrid_3d_bbox_loss


if __name__=="__main__":
    data_dir = "./data"
    epochs = 5
    dataset = Custom3DBBoxDataset(prepare_data(data_dir))
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    dl_model = MultiObject3DBBoxModel()
    optimizer = torch.optim.AdamW(dl_model.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-4)
    dl_model.train()

    loss_per_epoch = []
    for epoch in range(epochs):
        epoch_losses = 0.0
        print(f"Epoch: {epoch+1}")
        for i, batch in enumerate(train_dataloader):
            rgb, mask, pc_features, bbox3d = batch
            optimizer.zero_grad()
            pred_boxes = dl_model([rgb, mask, pc_features])
            loss = hybrid_3d_bbox_loss(pred_boxes, bbox3d)
            print(f"\tBatch: {i+1} ended with {loss} loss.")
            loss.backward()
            optimizer.step()
            epoch_losses+=loss

        print(f"Epoch: {epoch+1} ended with loss {epoch_losses/len(train_dataloader)}.")
        loss_per_epoch.append(epoch_losses.detach()/len(train_dataloader))
        
    with torch.no_grad():
        test_loss = 0
        dl_model.eval()
        for i, batch in enumerate(test_dataloader):
            rgb, mask, pc_features, bbox3d = batch
            pred_boxes = dl_model([rgb, mask, pc_features])
            loss = hybrid_3d_bbox_loss(pred_boxes, bbox3d)
            test_loss+=loss

        print(f"Model Evaluation on Test Set: Loss: {test_loss/len(test_dataloader)}.")

    torch.save(dl_model, 'model.pt')

    plot_losses(loss_per_epoch)
