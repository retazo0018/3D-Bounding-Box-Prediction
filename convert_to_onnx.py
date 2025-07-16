import torch

def convert_pt_to_onnx(pt_file_path, onnx_file_path, input_sample):
    model = torch.load(pt_file_path)
    model.eval() 

    # Convert the PyTorch model to ONNX
    torch.onnx.export(
            model,
            input_sample,                # tuple of inputs, e.g. (rgb, mask, pc)
            onnx_file_path,
            opset_version=14,            # latest stable opset
            input_names=['rgb', 'mask', 'pc'],
            output_names=['pred_boxes'],
            dynamic_axes={
                'rgb': {0: 'batch_size'},
                'mask': {0: 'batch_size'},
                'pc': {0: 'batch_size'},
                'pred_boxes': {0: 'batch_size'}
            }
        )
    print(f"Model successfully converted and saved to {onnx_file_path}")

if __name__=="__main__":
    convert_pt_to_onnx('model.pt', 'model.onnx', (torch.zeros([1, 3, 512, 512]), torch.zeros([1, 25, 512, 512]), torch.zeros([1, 3, 512, 512])))
