"""
Since python is a language for children and torch uses a metaclass for `torch.autograd.Function`,
we cannot enforce properly the interface of every operation.

Convention

class MarineX(torch.autograd.Function):

    tols =


    @staticmethod
    def forward(ctx, *inputs):
        ...

    @staticmethod
    def backward(ctx, *ouputs):
        ...

    @classmethod
    def validate_fwd(cls, max_examples):
        ...

    @classmethod
    def validate_bwd(cls, max_examples):
        ...
"""
