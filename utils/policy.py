##
#
# Policy class for handling policy-related operations, 
# such as loading, inferencing, and getting properties.
#
##

import numpy as np
import torch
import onnx
import onnxruntime as ort


############################################################################
# HELPERS
############################################################################

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


# inference with a torch policy
def policy_inference_torch(policy, input):
    
    # convert to torch tensor and add batch dimension
    input_tensor = torch.from_numpy(input).unsqueeze(0)

    # forward pass (no_grad disables autograd for faster inference)
    with torch.no_grad():
        action = policy(input_tensor).numpy().squeeze()

    return action

# inference with an onnx policy
def policy_inference_onnx(session, input):

    # get input name
    input_name = session.get_inputs()[0].name

    # forward pass
    input_array = input.reshape(1, -1).astype(np.float32)
    action = session.run(None, {input_name: input_array})[0].squeeze()

    return action


############################################################################
# POLICY CLASS
############################################################################

class Policy:
    """
    Generic control policy class that can handle both PyTorch and ONNX policies.
    """

    def __init__(self, policy_path):

        # load the policy
        self._load_policy(policy_path)

        # compute important properties of the policy
        self._get_policy_properties()

        
    # load a policy given the path 
    def _load_policy(self, policy_path):
        # torch file
        if "pt" in policy_path.lower():
            self.policy = torch.jit.load(policy_path)
            self.policy.eval()
            self._policy_type = "torch"
        # onnx file
        elif "onnx" in policy_path.lower():
            self.policy = onnx.load(policy_path)
            self._onnx_session = ort.InferenceSession(self.policy.SerializeToString())
            self._policy_type = "onnx"
        # incompatible file
        else:
            raise ValueError("Unsupported policy format. Please use .pt or .onnx files.")


    # get important properties of the policy
    def _get_policy_properties(self):
        
        # I/O sizes
        if self._policy_type == "torch":
            self.input_size, self.output_size = get_policy_io_size_torch(self.policy)
        elif self._policy_type == "onnx":
            self.input_size, self.output_size = get_policy_io_size_onnx(self.policy)
        

    # inference the policy given an input
    def inference(self, input):
        if self._policy_type == "torch":
            return policy_inference_torch(self.policy, input)
        elif self._policy_type == "onnx":
            return policy_inference_onnx(self._onnx_session, input)
