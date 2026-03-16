##
#
# Assortment of useful functions to get policy properties
#
##

import numpy as np
import torch
import onnx
import onnxruntime as ort

# load a policy given the path 
def load_policy(policy_path):

    # torch file
    if "pt" in policy_path.lower():
        policy = torch.jit.load(policy_path)
        policy.eval()
        policy_type = "torch"
    
    # onnx file
    elif "onnx" in policy_path.lower():
        policy = onnx.load(policy_path)
        policy_type = "onnx"
    else:
        raise ValueError("Unsupported policy format. Please use .pt or .onnx files.")

    return policy, policy_type


# get policy input and output sizes
def get_policy_io_size(policy, policy_type):
    if policy_type == "torch":
        return get_policy_io_size_torch(policy)
    elif policy_type == "onnx":
        return get_policy_io_size_onnx(policy)
    else:
        raise ValueError("Unsupported policy type. Please use 'torch' or 'onnx'.")

# get the input and output dimensions of a torch policy
def get_policy_io_size_torch(policy, max_input_size=1024):
    
    # set the policy to eval mode
    policy.eval()

    # input size search
    input_size = None
    for input_size in range(1, max_input_size + 1):
        try:
            dummy = torch.zeros(1, input_size)
            with torch.no_grad():
                policy(dummy)
            break
        except Exception:
            continue
    
    # query the output size
    with torch.no_grad():
        output = policy(torch.zeros(1, input_size))
    output_size = output.shape[-1]

    return input_size, output_size

# get the input and output dimensions of an onnx policy
def get_policy_io_size_onnx(policy):

    # input size from first input tensor
    input_shape = policy.graph.input[0].type.tensor_type.shape
    input_size = input_shape.dim[-1].dim_value

    # output size from first output tensor
    output_shape = policy.graph.output[0].type.tensor_type.shape
    output_size = output_shape.dim[-1].dim_value

    return input_size, output_size


# inference a policy given an input
def policy_inference(policy, policy_type, input):
    if policy_type == "torch":
        return policy_inference_torch(policy, input)
    elif policy_type == "onnx":
        return policy_inference_onnx(policy, input)
    else:
        raise ValueError("Unsupported policy type. Please use 'torch' or 'onnx'.")

# inference with a torch policy
def policy_inference_torch(policy, input):
    
    # convert to torch tensor and add batch dimension
    input_tensor = torch.from_numpy(input).unsqueeze(0)

    # forward pass (no_grad disables autograd for faster inference)
    with torch.no_grad():
        action = policy(input_tensor).numpy().squeeze()

    return action

# inference with an onnx policy
def policy_inference_onnx(policy, input):

    # create an inference session
    session = ort.InferenceSession(policy.SerializeToString())

    # get input name
    input_name = session.get_inputs()[0].name

    # forward pass
    input_array = input.reshape(1, -1).astype(np.float32)
    action = session.run(None, {input_name: input_array})[0].squeeze()

    return action

