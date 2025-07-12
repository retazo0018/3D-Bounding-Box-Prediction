import torch

def convert_pt_to_onnx(pt_file_path, onnx_file_path, input_sample):
    model = torch.load(pt_file_path)
    model.eval() 

    # Convert the PyTorch model to ONNX
    torch.onnx.export(
        model,               
        input_sample,        
        onnx_file_path)

    print(f"Model successfully converted and saved to {onnx_file_path}")

if __name__=="__main__":
    convert_pt_to_onnx('model.pt', 'model.onnx', [torch.zeros([1, 3, 512, 512]), torch.zeros([1, 25, 512, 512]), torch.zeros([1, 3, 512, 512])])
