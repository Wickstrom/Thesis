import torch
from torch.autograd import Function


class GuidedBackpropReLU(Function):

    def forward(self, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(torch.zeros(
                 input.size()).type_as(input), input, positive_mask)
        self.save_for_backward(input, output)

        return output

    def backward(self, grad_output):
        input, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(
                     input.size()).type_as(input), torch.addcmul(torch.zeros(
                             input.size()).type_as(input), grad_output,
                             positive_mask_1), positive_mask_2)
        return grad_input


class GuidedBackpropModule(torch.nn.Module):

    def forward(self, input):
        return GuidedBackpropReLU()(input)
