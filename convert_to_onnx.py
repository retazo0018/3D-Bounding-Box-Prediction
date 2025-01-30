import torch

'''
torch.onnx.errors.UnsupportedOperatorError: Exporting the operator 'aten::_upsample_bilinear2d_aa' to ONNX opset version 17 
is not supported. Please feel free to request support or submit a pull request on 
PyTorch GitHub: https://github.com/pytorch/pytorch/issues.
'''

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
    convert_pt_to_onnx('model.pt', 'model.onnx', [torch.zeros([1, 3, 480, 640]), torch.zeros([1, 480, 640]), torch.zeros([1, 480*640, 3])])
