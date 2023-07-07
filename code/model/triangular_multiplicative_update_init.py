# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#from torch_utils import *
from functools import partialmethod
from typing import Optional
import torch
import numpy as np
import torch.nn as nn
import math
from scipy.stats import truncnorm
import deepspeed
from functools import partial


def _prod(nums):
    out = 1
    for n in nums:
        out = out * n
    return out

def add(m1, m2, inplace):
    # The first operation in a checkpoint can't be in-place, but it's
    # nice to have in-place addition during inference. Thus...
    if(not inplace):
        m1 = m1 + m2
    else:
        m1 =m1+ m2
    return m1

def permute_final_dims(tensor, inds):
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])

def _calculate_fan(linear_weight_shape, fan="fan_in"):
    fan_out, fan_in = linear_weight_shape

    if fan == "fan_in":
        f = fan_in
    elif fan == "fan_out":
        f = fan_out
    elif fan == "fan_avg":
        f = (fan_in + fan_out) / 2
    else:
        raise ValueError("Invalid fan option")
    return f

def trunc_normal_init_(weights, scale=1.0, fan="fan_in"):
    shape = weights.shape
    f = _calculate_fan(shape, fan)
    scale = scale / max(1, f)
    a = -2
    b = 2
    std = math.sqrt(scale) / truncnorm.std(a=a, b=b, loc=0, scale=1)
    size = _prod(shape)
    samples = truncnorm.rvs(a=a, b=b, loc=0, scale=std, size=size)
    samples = np.reshape(samples, shape)
    with torch.no_grad():
        weights.copy_(torch.tensor(samples, device=weights.device))

def lecun_normal_init_(weights):
    trunc_normal_init_(weights, scale=1.0)


def he_normal_init_(weights):
    trunc_normal_init_(weights, scale=2.0)


def glorot_uniform_init_(weights):
    nn.init.xavier_uniform_(weights, gain=1)


def final_init_(weights):
    with torch.no_grad():
        weights.fill_(0.0)


def gating_init_(weights):
    with torch.no_grad():
        weights.fill_(0.0)


def normal_init_(weights):
    torch.nn.init.kaiming_normal_(weights, nonlinearity="linear")


def ipa_point_weights_init_(weights):
    with torch.no_grad():
        softplus_inverse_1 = 0.541324854612918
        weights.fill_(softplus_inverse_1)

class Linear(nn.Linear):
    """
    A Linear layer with built-in nonstandard initializations. Called just
    like torch.nn.Linear.

    Implements the initializers in 1.11.4, plus some additional ones found
    in the code.
    """

    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            bias: bool = True,
            init: str = "default",
            init_fn = None,
    ):
        """
        Args:
            in_dim:
                The final dimension of inputs to the layer
            out_dim:
                The final dimension of layer outputs
            bias:
                Whether to learn an additive bias. True by default
            init:
                The initializer to use. Choose from:

                "default": LeCun fan-in truncated normal initialization
                "relu": He initialization w/ truncated normal distribution
                "glorot": Fan-average Glorot uniform initialization
                "gating": Weights=0, Bias=1
                "normal": Normal initialization with std=1/sqrt(fan_in)
                "final": Weights=0, Bias=0

                Overridden by init_fn if the latter is not None.
            init_fn:
                A custom initializer taking weight and bias as inputs.
                Overrides init if not None.
        """
        super(Linear, self).__init__(in_dim, out_dim, bias=bias)

        if bias:
            with torch.no_grad():
                self.bias.fill_(0)

        with torch.no_grad():
            if init_fn is not None:
                init_fn(self.weight, self.bias)
            else:
                if init == "default":
                    lecun_normal_init_(self.weight)
                elif init == "relu":
                    he_normal_init_(self.weight)
                elif init == "glorot":
                    glorot_uniform_init_(self.weight)
                elif init == "gating":
                    gating_init_(self.weight)
                    if bias:
                        self.bias.fill_(1.0)
                elif init == "normal":
                    normal_init_(self.weight)
                elif init == "final":
                    final_init_(self.weight)
                else:
                    raise ValueError("Invalid init string.")


class TriangleMultiplicativeUpdate(nn.Module):
    """
    Implements Algorithms 11 and 12.
    """
    def __init__(self, c_z, c_hidden, _outgoing=True):
        """
        Args:
            c_z:
                Input channel dimension
            c:
                Hidden channel dimension
        """
        super(TriangleMultiplicativeUpdate, self).__init__()
        self.c_z = c_z
        self.c_hidden = c_hidden
        self._outgoing = _outgoing

        self.linear_a_p = Linear(self.c_z, self.c_hidden)
        self.linear_a_g = Linear(self.c_z, self.c_hidden, init="gating")
        self.linear_b_p = Linear(self.c_z, self.c_hidden)
        self.linear_b_g = Linear(self.c_z, self.c_hidden, init="gating")
        self.linear_g = Linear(self.c_z, self.c_z, init="relu")
        self.linear_z = Linear(self.c_hidden, self.c_z, init="final")

        self.layer_norm_in = nn.LayerNorm(self.c_z)
        self.layer_norm_out = nn.LayerNorm(self.c_hidden)

        self.sigmoid = nn.Sigmoid()

    def _combine_projections(self,
        a: torch.Tensor,
        b: torch.Tensor,
        _inplace_chunk_size: Optional[int] = None
    ) -> torch.Tensor:
        if(self._outgoing):
            a = permute_final_dims(a, (2, 0, 1))
            b = permute_final_dims(b, (2, 1, 0))
        else:
            a = permute_final_dims(a, (2, 1, 0))
            b = permute_final_dims(b,  (2, 0, 1))

        if(_inplace_chunk_size is not None):
            # To be replaced by torch vmap
            for i in range(0, a.shape[-3], _inplace_chunk_size):
                a_chunk = a[..., i: i + _inplace_chunk_size, :, :]
                b_chunk = b[..., i: i + _inplace_chunk_size, :, :]
                a[..., i: i + _inplace_chunk_size, :, :] = (
                    torch.matmul(
                        a_chunk,
                        b_chunk,
                    )
                )

            p = a
        else:
            p = torch.matmul(a, b)

        return permute_final_dims(p, (1, 2, 0))

       

    def forward(self, 
        z: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        _inplace_chunk_size: Optional[int] = 256,
    ) -> torch.Tensor:
        """
        Args:
            x:
                [*, N_res, N_res, C_z] input tensor
            mask:
                [*, N_res, N_res] input mask
        Returns:
            [*, N_res, N_res, C_z] output tensor
        """

        if mask is None:
            mask = z.new_ones(z.shape[:-1])

        mask = mask.unsqueeze(-1)
        
        z = self.layer_norm_in(z.clone())
        a = mask
        a = a * self.sigmoid(self.linear_a_g(z)) 
        a = a * self.linear_a_p(z)
        b = mask
        b = b * self.sigmoid(self.linear_b_g(z))
        b = b * self.linear_b_p(z)
        x = self._combine_projections(a, b)
        del a, b
        x = self.layer_norm_out(x.clone())

        x = self.linear_z(x)
        g = self.sigmoid(self.linear_g(z))
        x = x * g

        return x


class TriangleMultiplicationOutgoing(TriangleMultiplicativeUpdate):
    """
    Implements Algorithm 11.
    """
    __init__ = partialmethod(TriangleMultiplicativeUpdate.__init__, _outgoing=True)


class TriangleMultiplicationIncoming(TriangleMultiplicativeUpdate):
    """
    Implements Algorithm 12.
    """
    __init__ = partialmethod(TriangleMultiplicativeUpdate.__init__, _outgoing=False)


class Attention(nn.Module):
    """
    Standard multi-head attention using AlphaFold's default layer
    initialization. Allows multiple bias vectors.
    """

    def __init__(
            self,
            c_q: int,
            c_k: int,
            c_v: int,
            c_hidden: int,
            no_heads: int,
            gating: bool = True,
    ):
        """
        Args:
            c_q:
                Input dimension of query data
            c_k:
                Input dimension of key data
            c_v:
                Input dimension of value data
            c_hidden:
                Per-head hidden dimension
            no_heads:
                Number of attention heads
            gating:
                Whether the output should be gated using query data
        """
        super(Attention, self).__init__()

        self.c_q = c_q
        self.c_k = c_k
        self.c_v = c_v
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.gating = gating

        # DISCREPANCY: c_hidden is not the per-head channel dimension, as
        # stated in the supplement, but the overall channel dimension.

        self.linear_q = Linear(
            self.c_q, self.c_hidden * self.no_heads, bias=False, init="glorot"
        )
        self.linear_k = Linear(
            self.c_k, self.c_hidden * self.no_heads, bias=False, init="glorot"
        )
        self.linear_v = Linear(
            self.c_v, self.c_hidden * self.no_heads, bias=False, init="glorot"
        )
        self.linear_o = Linear(
            self.c_hidden * self.no_heads, self.c_q, init="final"
        )

        self.linear_g = None
        if self.gating:
            self.linear_g = Linear(
                self.c_q, self.c_hidden * self.no_heads, init="gating"
            )

        self.sigmoid = nn.Sigmoid()

    def _prep_qkv(self,
                  q_x: torch.Tensor,
                  kv_x: torch.Tensor
                  ):
        # [*, Q/K/V, H * C_hidden]
        q = self.linear_q(q_x)
        k = self.linear_k(kv_x)
        v = self.linear_v(kv_x)

        # [*, Q/K, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))
        k = k.view(k.shape[:-1] + (self.no_heads, -1))
        v = v.view(v.shape[:-1] + (self.no_heads, -1))

        # [*, H, Q/K, C_hidden]
        q = q.transpose(-2, -3)
        k = k.transpose(-2, -3)
        v = v.transpose(-2, -3)

        q =q/ math.sqrt(self.c_hidden)

        return q, k, v


    def forward(
            self,
            q_x: torch.Tensor,
            kv_x: torch.Tensor,
            biases = None,
    ) -> torch.Tensor:
        """
        Args:
            q_x:
                [*, Q, C_q] query data
            kv_x:
                [*, K, C_k] key data
            biases:
                List of biases that broadcast to [*, H, Q, K]
        Returns
            [*, Q, C_q] attention update
        """


        # [*, H, Q/K, C_hidden]
        q, k, v = self._prep_qkv(q_x, kv_x)

        # [*, Q, H, C_hidden]
        o = _attention(q, k, v)
        o = o.transpose(-2, -3)

        return o

def _attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    # [*, H, C_hidden, K]
    key = permute_final_dims(key, (1, 0))

    # [*, H, Q, K]
    a = torch.matmul(query, key)


    a = softmax_no_cast(a, -1)

    # [*, H, Q, C_hidden]
    a = torch.matmul(a, value)

    return a

@torch.jit.ignore
def softmax_no_cast(t: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
        Softmax, but without automatic casting to fp32 when the input is of
        type bfloat16
    """
    d = t.dtype
    if(d is torch.bfloat16 and not deepspeed.utils.is_initialized()):
        with torch.cuda.amp.autocast(enabled=False):
            s = torch.nn.functional.softmax(t, dim=dim)
    else:
        s = torch.nn.functional.softmax(t, dim=dim)

    return s





class PairTransition(nn.Module):
    """
    Implements Algorithm 15.
    """

    def __init__(self, c_z, n):
        """
        Args:
            c_z:
                Pair transition channel dimension
            n:
                Factor by which c_z is multiplied to obtain hidden channel
                dimension
        """
        super(PairTransition, self).__init__()

        self.c_z = c_z
        self.n = n
        self.layer_norm = nn.LayerNorm(self.c_z)
        self.linear_1 = Linear(self.c_z, self.n * self.c_z, init="relu")
        self.relu = nn.ReLU()
        self.linear_2 = Linear(self.n * self.c_z, c_z, init="final")

    def forward(self, z,mask = None):
        # [*, N_res, N_res, C_z]
        if mask is None:
            mask = z.new_ones(z.shape[:-1])

        # [*, N_res, N_res, 1]
        mask = mask.unsqueeze(-1)
        # print(z.shape[0],z.shape[1],z.shape[2],self.c_z * self.n)
        # z = z.reshape(z.shape[0],z.shape[1],z.shape[2],self.c_z * self.n)
        z = self.layer_norm(z.clone())

        # [*, N_res, N_res, C_hidden]
        z = self.linear_1(z)
        z = self.relu(z)

        # [*, N_res, N_res, C_z]
        z = self.linear_2(z) * mask

        return z

def relpos(ri: torch.Tensor):
    """
    Computes relative positional encodings

    Implements Algorithm 4.

    Args:
        ri:
            "residue_index" features of shape [*, N]
    """
    d = ri[..., None] - ri[..., None, :]
    boundaries = torch.arange(
        start=-32, end=32 + 1, device=d.device
    )
    reshaped_bins = boundaries.view(((1,) * len(d.shape)) + (len(boundaries),))
    d = d[..., None] - reshaped_bins
    d = torch.abs(d)
    d = torch.argmin(d, dim=-1)
    d = nn.functional.one_hot(d, num_classes=len(boundaries)).float()
    d = d.to(ri.dtype)
    return d


if __name__ == '__main__':
    pair = np.load('../feats/7o33_pair.npy')
    pair = torch.tensor(pair,dtype=torch.float).unsqueeze(0)
    print(pair.shape)
    a = TriangleAttentionStartingNode( 60,120,no_heads=4)
    c = TriangleAttentionEndingNode( 60,120,no_heads=4)
    b = PairTransition(60,2)
    new_pair = a(torch.tensor(pair,dtype=torch.float))
    print(new_pair.shape)
    d = c(new_pair)
    print(d.shape)
    e = b._transition(d,None)
    print(e.shape)
