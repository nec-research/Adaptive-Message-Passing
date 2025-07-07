#   Adaptive Message Passing
	
#   Authors: Federico Errica (Federico.Errica@neclab.eu) 
#            Henrik Christiansen (Henrik.Christiansen@neclab.eu)
# 	    Viktor Zaverkin (Viktor.Zaverkin@neclab.eu)
#   	    Takashi Maruyama (Takashi.Maruyama@neclab.eu)
#  	    Mathias Niepert (mathias.niepert@ki.uni-stuttgart.de)
#  	    Francesco Alesiani (Francesco.Alesiani @neclab.eu)
  
#   Files:    
#             distribution.py, 
#             layer_generator.py, 
#             model.py, 
#             util.py,
#             example.py 
            
# NEC Laboratories Europe GmbH, Copyright (c) 2025-, All rights reserved.  

#        THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
 
#        PROPRIETARY INFORMATION ---  

# SOFTWARE LICENSE AGREEMENT

# ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY

# BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS
# LICENSE AGREEMENT.  IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR
# DOWNLOAD THE SOFTWARE.

# This is a license agreement ("Agreement") between your academic institution
# or non-profit organization or self (called "Licensee" or "You" in this
# Agreement) and NEC Laboratories Europe GmbH (called "Licensor" in this
# Agreement).  All rights not specifically granted to you in this Agreement
# are reserved for Licensor. 

# RESERVATION OF OWNERSHIP AND GRANT OF LICENSE: Licensor retains exclusive
# ownership of any copy of the Software (as defined below) licensed under this
# Agreement and hereby grants to Licensee a personal, non-exclusive,
# non-transferable license to use the Software for noncommercial research
# purposes, without the right to sublicense, pursuant to the terms and
# conditions of this Agreement. NO EXPRESS OR IMPLIED LICENSES TO ANY OF
# LICENSOR'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE. As used in this
# Agreement, the term "Software" means (i) the actual copy of all or any
# portion of code for program routines made accessible to Licensee by Licensor
# pursuant to this Agreement, inclusive of backups, updates, and/or merged
# copies permitted hereunder or subsequently supplied by Licensor,  including
# all or any file structures, programming instructions, user interfaces and
# screen formats and sequences as well as any and all documentation and
# instructions related to it, and (ii) all or any derivatives and/or
# modifications created or made by You to any of the items specified in (i).

# CONFIDENTIALITY/PUBLICATIONS: Licensee acknowledges that the Software is
# proprietary to Licensor, and as such, Licensee agrees to receive all such
# materials and to use the Software only in accordance with the terms of this
# Agreement.  Licensee agrees to use reasonable effort to protect the Software
# from unauthorized use, reproduction, distribution, or publication. All
# publication materials mentioning features or use of this software must
# explicitly include an acknowledgement the software was developed by NEC
# Laboratories Europe GmbH.

# COPYRIGHT: The Software is owned by Licensor.  

# PERMITTED USES:  The Software may be used for your own noncommercial
# internal research purposes. You understand and agree that Licensor is not
# obligated to implement any suggestions and/or feedback you might provide
# regarding the Software, but to the extent Licensor does so, you are not
# entitled to any compensation related thereto.

# DERIVATIVES: You may create derivatives of or make modifications to the
# Software, however, You agree that all and any such derivatives and
# modifications will be owned by Licensor and become a part of the Software
# licensed to You under this Agreement.  You may only use such derivatives and
# modifications for your own noncommercial internal research purposes, and you
# may not otherwise use, distribute or copy such derivatives and modifications
# in violation of this Agreement.

# BACKUPS:  If Licensee is an organization, it may make that number of copies
# of the Software necessary for internal noncommercial use at a single site
# within its organization provided that all information appearing in or on the
# original labels, including the copyright and trademark notices are copied
# onto the labels of the copies.

# USES NOT PERMITTED:  You may not distribute, copy or use the Software except
# as explicitly permitted herein. Licensee has not been granted any trademark
# license as part of this Agreement.  Neither the name of NEC Laboratories
# Europe GmbH nor the names of its contributors may be used to endorse or
# promote products derived from this Software without specific prior written
# permission.

# You may not sell, rent, lease, sublicense, lend, time-share or transfer, in
# whole or in part, or provide third parties access to prior or present
# versions (or any parts thereof) of the Software.

# ASSIGNMENT: You may not assign this Agreement or your rights hereunder
# without the prior written consent of Licensor. Any attempted assignment
# without such consent shall be null and void.

# TERM: The term of the license granted by this Agreement is from Licensee's
# acceptance of this Agreement by downloading the Software or by using the
# Software until terminated as provided below.  

# The Agreement automatically terminates without notice if you fail to comply
# with any provision of this Agreement.  Licensee may terminate this Agreement
# by ceasing using the Software.  Upon any termination of this Agreement,
# Licensee will delete any and all copies of the Software. You agree that all
# provisions which operate to protect the proprietary rights of Licensor shall
# remain in force should breach occur and that the obligation of
# confidentiality described in this Agreement is binding in perpetuity and, as
# such, survives the term of the Agreement.

# FEE: Provided Licensee abides completely by the terms and conditions of this
# Agreement, there is no fee due to Licensor for Licensee's use of the
# Software in accordance with this Agreement.

# DISCLAIMER OF WARRANTIES:  THE SOFTWARE IS PROVIDED "AS-IS" WITHOUT WARRANTY
# OF ANY KIND INCLUDING ANY WARRANTIES OF PERFORMANCE OR MERCHANTABILITY OR
# FITNESS FOR A PARTICULAR USE OR PURPOSE OR OF NON- INFRINGEMENT.  LICENSEE
# BEARS ALL RISK RELATING TO QUALITY AND PERFORMANCE OF THE SOFTWARE AND
# RELATED MATERIALS.

# SUPPORT AND MAINTENANCE: No Software support or training by the Licensor is
# provided as part of this Agreement.  

# EXCLUSIVE REMEDY AND LIMITATION OF LIABILITY: To the maximum extent
# permitted under applicable law, Licensor shall not be liable for direct,
# indirect, special, incidental, or consequential damages or lost profits
# related to Licensee's use of and/or inability to use the Software, even if
# Licensor is advised of the possibility of such damage.

# EXPORT REGULATION: Licensee agrees to comply with any and all applicable
# export control laws, regulations, and/or other laws related to embargoes and
# sanction programs administered by law.

# SEVERABILITY: If any provision(s) of this Agreement shall be held to be
# invalid, illegal, or unenforceable by a court or other tribunal of competent
# jurisdiction, the validity, legality and enforceability of the remaining
# provisions shall not in any way be affected or impaired thereby.

# NO IMPLIED WAIVERS: No failure or delay by Licensor in enforcing any right
# or remedy under this Agreement shall be construed as a waiver of any future
# or other exercise of such right or remedy by Licensor.

# GOVERNING LAW: This Agreement shall be construed and enforced in accordance
# with the laws of Germany without reference to conflict of laws principles.
# You consent to the personal jurisdiction of the courts of this country and
# waive their rights to venue outside of Germany.

# ENTIRE AGREEMENT AND AMENDMENTS: This Agreement constitutes the sole and
# entire agreement between Licensee and Licensor as to the matter set forth
# herein and supersedes any previous agreements, understandings, and
# arrangements between the parties relating hereto.

import math
from typing import Tuple, List

import scipy.stats as st
import torch
from torch import inf
from torch.distributions import Normal, Poisson as tPoisson
from torch.nn import Parameter, Module, ModuleList
from torch.nn.functional import softplus

from util import return_class_and_args


def softplus_inverse(x):
    """log(exp(x) - 1)"""
    return torch.where(x > 10, x, x.expm1().log())


class ContinuousDistribution(Module):
    """
    Implements an interface for this package
    """

    def __init__(self):
        super().__init__()
        self.device = None

    def to(self, device):
        super().to(device)
        self.device = device

    def _validate_args(self, value):
        assert isinstance(
            value, torch.Tensor
        ), f"expected torch tensor, found {type(value)}"

        assert isinstance(value, torch.FloatTensor) or (
            value.dtype == torch.float32
        ), f"expected float tensor, found {value.dtype}"

        assert len(value.shape) == 2, f"expected shape: (N,1), found {value.shape}"

        assert (
            value.shape[1] == 1
        ), f"expected one-dimensional values, found {value.shape}"

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """
        Computes the log pdf of the distribution

        :param value: a tensor of shape Nx1, where N is the number of samples

        :return: a tensor of shape Nx1
        """
        raise NotImplementedError(
            "You should subclass Distribution and " "implement this method."
        )

    def cdf(self, value):
        """
        Computes the cdf of the distribution

        :param value: a tensor of shape Nx1, where N is the number of samples

        :return: a tensor of shape Nx1
        """
        raise NotImplementedError(
            "You should subclass Distribution and " "implement this method."
        )

    def quantile(self, p: float = 0.99) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the p-quantile for the distribution.

        :param p: the parameter p of the quantile

        :return: lower and upper bounds for the p-quantile. If the p-quantile
            can be computed exactly then they are the same

        """
        raise NotImplementedError(
            "You should subclass Distribution and " "implement this method."
        )


class FoldedNormal(ContinuousDistribution):
    def __init__(self, loc: float, scale: float):
        """
        Creates a folded-normal distribution parameterized by `mean` and `scale`
        where

            X ~ Normal(mean, scale)
            Y = |X| ~ FoldedNormal(scale)

        :param mean: the mean
        :param scale: the standard deviation
        """
        super(FoldedNormal, self).__init__()

        assert loc >= 0.0, (
            "expected loc >=0 for our work and for a correct quantile" " computation"
        )
        self.base_loc = Parameter(torch.tensor([loc]), requires_grad=True)
        self._base_scale = Parameter(torch.tensor([scale]), requires_grad=True)

    def get_q_ell_named_parameters(self) -> dict:
        return {
            "folded_normal_mean": self.mean,
            "folded_normal_variance": self.variance,
        }

    @property
    def base_scale(self) -> torch.Tensor:
        # prevent collapse to zero variance
        const = torch.Tensor([0.5]).to(self.device)

        return softplus(self._base_scale) + const

    @property
    def mean(self) -> torch.Tensor:
        """
        Computes the mean of the folded normal distribution

        :return: the mean as a torch.Tensor
        """
        half = torch.Tensor([0.5]).to(self.device)
        one = torch.Tensor([1.0]).to(self.device)
        two = torch.Tensor([2.0]).to(self.device)
        pi = torch.Tensor([math.pi]).to(self.device)

        mu_squared = torch.pow(self.base_loc, two)
        sigma_squared = torch.pow(self.base_scale, two)

        mean = self.base_scale * torch.sqrt(two / pi) * torch.exp(
            -half * (mu_squared / sigma_squared)
        ) + self.base_loc * (
            one
            - two
            * half
            * (
                one
                + torch.erf(
                    -torch.Tensor([self.base_loc]) / torch.tensor([self.base_scale])
                ).to(self.device)
            )
        )
        return mean

    @property
    def variance(self) -> torch.Tensor:
        """
        Computes the variance of the folded normal distribution

        :return: the variance as a torch.Tensor
        """
        two = torch.Tensor([2.0]).to(self.device)
        mu_squared = torch.pow(self.base_loc, two)
        sigma_squared = torch.pow(self.base_scale, two)

        # add a minimal variance to avoid degenerate cases
        return mu_squared + sigma_squared - torch.pow(self.mean, two)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """
        Computes the log pdf of the distribution

        :param value: a tensor of shape Nx1, where N is the number of samples

        :return: a tensor of shape Nx1
        """
        self._validate_args(value)

        base_dist_1 = Normal(self.base_loc, self.base_scale)
        base_dist_2 = Normal(-self.base_loc, self.base_scale)

        log_prob = torch.logsumexp(
            torch.stack(
                (
                    base_dist_1.log_prob(value),
                    base_dist_2.log_prob(value),
                ),
                dim=-1,
            ),
            dim=-1,
        )

        # deal with cases where x <= 0
        log_prob = torch.where(value >= 0, log_prob, -inf)

        return log_prob

    def cdf(self, value):
        """
        Computes the cdf of the distribution

        :param value: a tensor of shape Nx1, where N is the number of samples

        :return: a tensor of shape Nx1
        """
        self._validate_args(value)

        base_dist_1 = Normal(self.base_loc + 1e-4, self.base_scale)
        base_dist_2 = Normal(-self.base_loc - 1e-4, self.base_scale)

        term_1 = (2.0 * base_dist_1.cdf(value)) - 1.0
        term_2 = (2.0 * base_dist_2.cdf(value)) - 1.0
        cdf = 0.5 * (term_1 + term_2)

        # deal with cases where x <= 0
        cdf = torch.where(value >= 0, cdf, 0)

        return cdf

    def _quantile_lower_bound(self, p: float = 0.99) -> torch.Tensor:
        # since cdf of normal always >= cdf folded normal, any p-quantile of
        # normal is <= p-quantile of the folded normal. Hence use as lower
        # bound
        p = torch.tensor([p])
        mu = torch.tensor([self.base_loc], device="cpu")
        sigma = torch.tensor([self.base_scale], device="cpu")
        sqrt_two = torch.sqrt(torch.tensor([2.0]))
        normal_quantile = mu + sigma * sqrt_two * torch.erfinv(2.0 * p - 1.0)

        # if normal quantile is x < 0, then it becomes x'=0 in a folded normal
        # but we require mu > 0 so it should no t be a problem
        return torch.relu(normal_quantile)

    def _quantile_upper_bound(self, p: float = 0.99) -> torch.Tensor:
        # upper bound derived from Chernoff's bound with t = 1/sigma

        def normal_cdf(x):
            return 0.5 * (1 + torch.erf(x / torch.sqrt(torch.tensor([2.0]))))

        mu = torch.tensor([self.base_loc], device="cpu")
        sigma = torch.tensor([self.base_scale], device="cpu")
        mu_by_sigma = mu / sigma

        remainder = torch.tensor([1.0 - p])
        const = torch.exp(torch.tensor([0.5])) * (
            normal_cdf(1 + mu_by_sigma)
            + (normal_cdf(1 - mu_by_sigma) / torch.exp(2 * mu_by_sigma))
        )
        p_quantile_upper_bound = (
            mu + sigma * torch.log(const) - sigma * torch.log(remainder)
        )
        return p_quantile_upper_bound

    def quantile(self, p: float = 0.99) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes an approximation of the p-quantile of a folded normal
            distribution

        :param p: the parameter p of the quantile

        :return: lower and upper bounds for the p-quantile. If the p-quantile
            can be computed exactly then they are the same
        """
        assert isinstance(p, float), "expected p argument of type float"
        return self._quantile_lower_bound(p), self._quantile_upper_bound(p)


class DiscretizedDistribution(Module):
    def __init__(self, **kwargs):
        """
        Creates a discretized version of a continuous distribution such that

            p(x) = phi(x+1) - phi(x)

        where phi is the cdf of the original distribution.

        :param kwargs: a dictionary with a key 'base_distribution' that
            allows us to instantiate a discretized distribution
        """
        super().__init__()
        base_d_cls, base_d_args = return_class_and_args(kwargs, "base_distribution")
        self.base_distribution = base_d_cls(**base_d_args)

    def to(self, device):
        super().to(device)
        self.device = device
        self.base_distribution.to(device)

    def get_q_ell_named_parameters(self) -> dict:
        return self.base_distribution.get_q_ell_named_parameters()

    def _validate_args(self, value):
        self.base_distribution._validate_args(value)

        # check values are integers
        assert torch.allclose(
            value, value.int().float()
        ), f"expected float tensor with integer values, got {value, self.base_distribution.mean}."

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """
        Computes the log pdf of the distribution

        :param value: a tensor of shape Nx1, where N is the number of samples

        :return: a tensor of shape Nx1
        """
        self._validate_args(value)

        one = torch.ones(1).to(value.device)
        # avoids a degenerate case where the base distribution has the
        # same cdf for both value and value+1, which leads to nan.
        tmp = torch.ones(1).to(value.device) * 1e-3

        return torch.log(
            self.base_distribution.cdf(value + one)
            - self.base_distribution.cdf(value)
            + tmp
        )

    def cdf(self, value: torch.Tensor) -> torch.Tensor:
        """
        Computes the cdf of the distribution

        :param value: a tensor of shape Nx1, where N is the number of samples

        :return: a tensor of shape Nx1
        """
        self._validate_args(value)
        one = torch.ones(1).to(value.device)

        return self.base_distribution.cdf(value + one)

    def quantile(self, p: float = 0.99) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the approximated p-quantile for the discrete distribution.
        The lower and upper bounds returned by the method will coincide, since
        we provide the smallest integer x such that cdf(x) >= p

        :param p: the parameter p

        :return: lower and upper bounds for the p-quantile. If the p-quantile
            can be computed exactly then they are the same
        """
        lower_bound, upper_bound = self.base_distribution.quantile(p)

        # Now perform binary search over the integers to find the smallest x
        # such that cdf(x) >= p. The boundaries of the search are given by the
        # bounds, and we use the fact that the cdf forms an ordered sequence
        l = torch.floor(lower_bound).to(self.device)
        u = torch.ceil(upper_bound).to(self.device)

        if self.cdf((u).unsqueeze(1)) < p:
            ok = False
            while not ok:
                u += 1
                if self.cdf((u).unsqueeze(1)) >= p:
                    ok = True
            return u, u

        # corner case
        if l == u:
            assert self.cdf(l.unsqueeze(1)) >= p
            return u, u

        # if lower bound is already sufficient, stop, the normal and folded
        # normal curves are very similar at the desired quantile
        if self.cdf(l.unsqueeze(1)) >= p:
            assert self.cdf(l.unsqueeze(1)) >= p
            return l, l

        # adapt the search: U will always have cdf(U) > p, so we need to
        # check when we move from cdf(U) > p to cdf(U-1) <= p
        while l <= u:
            if l == (u - 1.0) or (l == u):
                assert self.cdf((u + 1).unsqueeze(1)) >= p
                return u + 1, u + 1

            m = torch.floor((l + u) / 2.0)
            cdf_m = self.cdf(m.unsqueeze(1))

            if cdf_m < p:
                # move L to the right, closing the gap
                l = m + 1.0
            elif cdf_m > p:
                # move U to the left, closing the gap
                u = m - 1.0

    def compute_probability_vector(self, x) -> torch.Tensor:
        """
        Computes the **renormalized** vector of probabilities on the fly

        :return: a vector of arbitrary length with the probabilities
        """
        log_probs = self.log_prob(x).squeeze(1)
        probs = log_probs.exp()
        probs = probs / probs.sum()
        return probs

    @property
    def mean(self) -> torch.Tensor:
        return self.base_distribution.mean

    @property
    def variance(self) -> torch.Tensor:
        return self.base_distribution.variance


class Poisson(Module):
    """
    Implements a wrapper around the Poisson distribution
    """

    def __init__(self, rate: float):
        super().__init__()
        self.rate = Parameter(
            (softplus_inverse(torch.tensor([rate]))), requires_grad=True
        )

    def to(self, device):
        super().to(device)
        self.device = device

    def get_q_ell_named_parameters(self) -> dict:
        return {"poisson_mean": self.mean, "poisson_variance": self.mean}

    def _validate_args(self, value):
        # check values are integers
        assert torch.allclose(
            value, value.int().float()
        ), f"expected float tensor with integer values, got {value}."

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        self._validate_args(value)
        return tPoisson(rate=self.mean).log_prob(value)

    def cdf(self, value):
        # WARNING: no gradient will flow here, need a different
        # discretized distribution implementation
        self._validate_args(value)
        p = st.poisson(self.mean.item())
        return (
            torch.tensor(p.cdf(value.detach().cpu().numpy())).float().to(value.device)
        )

    def quantile(self, p: float = 0.99) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns Lower and Upper bounds as computed in ICML 2022 paper
        """
        rate = self.mean

        if p != 0.99:
            ub_quantile = torch.ceil(torch.tensor([10000.0]))
        else:
            ub_quantile = torch.ceil(1.3 * rate + 5.0)

        # assert p == 0.99, "Upper bound to poisson available only for 0.99"

        lb_quantile = torch.floor(rate - torch.log(torch.tensor([2.0])).to(rate.device))

        if lb_quantile < 0.0:
            lb_quantile = torch.zeros(1)

        for i in range(int(lb_quantile), int(ub_quantile) + 1):
            x = torch.tensor([i]).float()
            cmf = self.cdf(x)
            if cmf >= p:
                return x + 1, x + 1
        raise Exception("Quantile not found, check arguments.")

    def compute_probability_vector(self, x) -> torch.Tensor:
        """
        Computes the **renormalized** vector of probabilities on the fly

        :return: a vector of arbitrary length with the probabilities
        """
        alpha_L = (x * self.mean.log() - torch.lgamma(x + 1)).exp().squeeze(1)
        probs = alpha_L / alpha_L.sum()
        return probs

    @property
    def mean(self) -> torch.Tensor:
        return softplus(self.rate)

    @property
    def variance(self) -> torch.Tensor:
        return self.mean


class TruncatedDistribution(Module):
    def __init__(self, truncation_quantile: float, **kwargs):
        """
        Truncates a discretized distribution to a given quantile and
        renormalizes its probability.

        :param truncation_quantile: the quantile in [0,1] at which we want
            to truncate the discrete distribution.
        :param kwargs: a dictionary with a key 'discretized_distribution' that
            allows us to instantiate a discretized distribution
        """
        super().__init__()

        dist_d_cls, dist_d_args = return_class_and_args(
            kwargs, "discretized_distribution"
        )

        self.discretized_distribution = dist_d_cls(**dist_d_args)
        self.truncation_quantile = truncation_quantile

    def to(self, device):
        super().to(device)
        self.device = device
        self.discretized_distribution.to(device)

    def get_q_ell_named_parameters(self) -> dict:
        return self.discretized_distribution.get_q_ell_named_parameters()

    def compute_truncation_number(self) -> int:
        """
        Computes the truncation number at the specified quantile.

        :return: a positive integer holding the truncation number

        """

        # exploits the implementation of quantile() for the
        # DiscretizedDistribution, which returns
        _, truncation_number = self.discretized_distribution.quantile(
            p=self.truncation_quantile
        )

        # detach: this must not be part of the gradient computation in any way
        truncation_number = int(truncation_number.detach())

        assert truncation_number > 0
        return truncation_number

    def compute_probability_vector(self) -> torch.Tensor:
        """
        Computes the **renormalized** vector of probabilities on the fly

        :return: a vector of arbitrary length with the probabilities
        """
        truncation_number = self.compute_truncation_number()

        x = torch.arange(
            truncation_number + 1, dtype=torch.float32, device=self.device
        ).unsqueeze(1)

        probs = self.discretized_distribution.compute_probability_vector(x)

        assert torch.allclose(probs.sum(), torch.ones(1, device=self.device))
        return probs

    @property
    def mean(self) -> torch.Tensor:
        proba = self.compute_probability_vector()
        return (proba * torch.arange(len(proba)).to(proba.device)).sum()


class MixtureTruncated(Module):
    def __init__(
        self,
        truncation_quantile: float,
        distribution_list: List[dict],
    ):
        super().__init__()

        self.truncation_quantile = truncation_quantile

        # Randomly initialize mixing weights
        self.num_mixtures = len(distribution_list)
        mixing_weights = torch.rand(self.num_mixtures)
        self._mixing_weights = Parameter(mixing_weights, requires_grad=True)

        # instantiate list of distributions as defined
        d_list = []
        for d in distribution_list:
            for k, v in d.items():
                if "discretized_distribution" in k:
                    d_cls, d_args = return_class_and_args(d, k)
                    d_list.append(d_cls(**d_args))
        self.distributions = ModuleList(d_list)

        self.device = None

    def to(self, device):
        super().to(device)
        self.device = device
        for d in self.distributions:
            d.to(device)

    def get_q_ell_named_parameters(self) -> dict:
        named_params = {}

        for i, d in enumerate(self.distributions):
            d_named_params = d.get_q_ell_named_parameters()

            for k, v in d_named_params.items():
                named_params[f"{k}_{i+i}"] = v

        named_params["mixture_mixing_weights"] = self.mixing_weights

        return named_params

    @property
    def mixing_weights(self):
        eps = torch.Tensor([1e-4]).to(self.device)

        # normalize mixing weights
        mw = softplus(self._mixing_weights) + eps
        return mw / mw.sum()

    def compute_truncation_number(self) -> int:
        """
        Computes the truncation number at the specified quantile.

        :return: a positive integer holding the truncation number

        """

        # exploits the implementation of quantile() for the
        # DiscretizedDistribution, which returns
        truncation_list = [
            d.quantile(p=self.truncation_quantile)[1] for d in self.distributions
        ]
        truncation_number = max(truncation_list)

        # detach: this must not be part of the gradient computation in any way
        truncation_number = int(truncation_number.detach())

        assert truncation_number > 0
        return truncation_number

    def compute_probability_vector(self) -> torch.Tensor:
        """
        Computes the **renormalized** vector of probabilities on the fly

        :return: a vector of arbitrary length with the probabilities
        """
        truncation_number = self.compute_truncation_number()

        # no gradient so far, we detach on purpose
        # +1 because we need to have p(truncation_number) in the vector
        x = torch.arange(
            truncation_number + 1, dtype=torch.float, device=self.device
        ).unsqueeze(1)

        # gradient starts flowing from here!
        unnorm_log_probs = torch.cat([d.log_prob(x) for d in self.distributions], dim=1)
        w_unnorm_log_probs = unnorm_log_probs + self.mixing_weights.log().unsqueeze(0)
        w_unnorm_log_probs = torch.logsumexp(w_unnorm_log_probs, dim=1)

        # renormalize using log-sum-exp trick
        # norm_log_probs = w_unnorm_log_probs - torch.logsumexp(
        #     w_unnorm_log_probs, dim=0
        # )
        # assert torch.allclose(
        #     norm_log_probs.exp().sum(), torch.ones(1, device=self.device)
        # )
        # return norm_log_probs
        probs = w_unnorm_log_probs.exp()
        probs = probs / probs.sum()

        assert torch.allclose(probs.sum(), torch.ones(1, device=self.device)), probs
        return probs
