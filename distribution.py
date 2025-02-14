import math
from typing import Tuple, List

import scipy.stats as st

import torch
from pydgn.evaluation.util import return_class_and_args
from torch import inf
from torch.distributions import Normal, Poisson as tPoisson
from torch.nn import Parameter, Module, ModuleList
from torch.nn.functional import softplus


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

        assert (
            len(value.shape) == 2
        ), f"expected shape: (N,1), found {value.shape}"

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

    def quantile(self, p: float = 0.95) -> Tuple[torch.Tensor, torch.Tensor]:
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
            "expected loc >=0 for our work and for a correct quantile"
            " computation"
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
                    -torch.Tensor([self.base_loc])
                    / torch.tensor([self.base_scale])
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

    def _quantile_lower_bound(self, p: float = 0.95) -> torch.Tensor:
        # since cdf of normal always >= cdf folded normal, any p-quantile of
        # normal is <= p-quantile of the folded normal. Hence use as lower
        # bound
        p = torch.tensor([p])
        mu = torch.tensor([self.base_loc], device="cpu")
        sigma = torch.tensor([self.base_scale], device="cpu")
        sqrt_two = torch.sqrt(torch.tensor([2.0]))
        normal_quantile = mu + sigma * sqrt_two * torch.erfinv(2.0 * p - 1.0)

        # if normal quantile is x < 0, then it becomes x'=0 in a folded normal
        # but we require mu > 0 so it should not be a problem
        return torch.relu(normal_quantile)

    def _quantile_upper_bound(self, p: float = 0.95) -> torch.Tensor:
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

    def quantile(self, p: float = 0.95) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes an approximation of the p-quantile of a folded normal
            distribution

        :param p: the parameter p of the quantile

        :return: lower and upper bounds for the p-quantile. If the p-quantile
            can be computed exactly then they are the same
        """
        assert isinstance(p, float), "expected p argument of type float"
        return self._quantile_lower_bound(p), self._quantile_upper_bound(p)


def test_folded_normal_1():
    """
    Test mean, variance, pdf and cdf of folded normal for half normal case
    """
    two = torch.Tensor([2.0])
    pi = torch.Tensor([math.pi])

    n = Normal(loc=0, scale=1.0)
    fn = FoldedNormal(loc=0, scale=1.0)

    # test mean
    assert torch.isclose(fn.mean, torch.sqrt(two / pi))

    # test variance
    assert torch.isclose(fn.variance, 1 - torch.pow(fn.mean, two))

    # test pdf
    x = torch.tensor([0.0]).unsqueeze(1)
    assert fn.log_prob(x).exp() == 2.0 * n.log_prob(x).exp()

    # test cfg
    assert torch.isclose(
        fn.cdf(torch.zeros(1, 1)),
        torch.erf(torch.zeros(1) / (fn.base_scale * torch.sqrt(two))),
    )


def test_folded_normal_2():
    """
    Testing _validate_args() works as expected for half normal case
    """
    fn = FoldedNormal(loc=0, scale=1.0)

    ok = False
    try:
        fn.log_prob(0.0)
    except AssertionError as e:
        ok = True
    assert ok

    ok = False
    try:
        x = torch.zeros(1)
        assert not fn.log_prob(x)

        x = torch.zeros(5)
        assert not fn.log_prob(x)
    except AssertionError as e:
        ok = True
    assert ok

    ok = False
    try:
        x = torch.zeros(4, 6)
        assert not fn.log_prob(x)

        x = torch.zeros(1, 2)
        assert not fn.log_prob(x)
    except AssertionError as e:
        ok = True
    assert ok


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
        base_d_cls, base_d_args = return_class_and_args(
            kwargs, "base_distribution"
        )
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
        # same cdf for both value and value+1
        # which leads to nan. Also, a too small value can cause some
        # distributions to have prob 1 for a single layer, and the model
        # gets trapped in there
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

    def quantile(self, p: float = 0.95) -> Tuple[torch.Tensor, torch.Tensor]:
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

        # ------------------------------------------------------------------ #
        # TODO fix this! issues with binary search when using quantile = 0.99)
        #  we could also remove the binary search
        if self.cdf((u).unsqueeze(1)) < p:
            ok = False
            while not ok:
                u += 1
                if self.cdf((u).unsqueeze(1)) >= p:
                    ok = True
            return u,u
        # ------------------------------------------------------------------ #

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
            torch.tensor(p.cdf(value.detach().cpu().numpy()))
            .float()
            .to(value.device)
        )

    def quantile(self, p: float = 0.95) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns Lower and Upper bounds as computed in ICML 2022 paper
        """
        rate = self.mean

        if p != 0.95:
            ub_quantile = torch.ceil(torch.tensor([10000.]))
        else:
            ub_quantile = torch.ceil(1.3 * rate + 5.0)

        # assert p == 0.95, "Upper bound to poisson available only for 0.95"


        lb_quantile = torch.floor(
            rate - torch.log(torch.tensor([2.0])).to(rate.device)
        )

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

        # no gradient so far, we detach on purpose
        # +1 because we need to have p(truncation_number) in the vector
        x = torch.arange(
            truncation_number + 1, dtype=torch.float32, device=self.device
        ).unsqueeze(1)

        probs = self.discretized_distribution.compute_probability_vector(x)

        # gradient starts flowing from here!
        # unnorm_log_probs = self.discretized_distribution.log_prob(x).squeeze(1)

        # print(f'sum probs: {unnorm_log_probs.exp().sum()}')
        # print(f'sum probs minus last: {unnorm_log_probs.exp()[:-1].sum()}')

        # # renormalize using log-sum-exp trick
        # norm_log_probs = unnorm_log_probs - torch.logsumexp(
        #     unnorm_log_probs, dim=0
        # )
        # assert torch.allclose(
        #     norm_log_probs.exp().sum(), torch.ones(1, device=self.device)
        # )
        # return norm_log_probs

        # FIXME this is what happens in the original paper
        #  we want to learn the parameters for first layer as well, so
        #  we commented below and added a + 1 to truncation number
        # probs = torch.cat([torch.zeros(1,
        #                                device=probs.device,
        #                                dtype=probs.dtype),
        #                    probs])

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
            d.quantile(p=self.truncation_quantile)[1]
            for d in self.distributions
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
        unnorm_log_probs = torch.cat(
            [d.log_prob(x) for d in self.distributions], dim=1
        )
        w_unnorm_log_probs = (
            unnorm_log_probs + self.mixing_weights.log().unsqueeze(0)
        )
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

        # FIXME this is what happens in the original paper
        #  we want to learn the parameters for first layer as well, so
        #  we commented below and added a + 1 to truncation number
        # probs = torch.cat([torch.zeros(1,
        #                                device=probs.device,
        #                                dtype=probs.dtype),
        #                    probs])

        assert torch.allclose(probs.sum(), torch.ones(1, device=self.device)), probs
        return probs


class FixedDepth(Module):
    def __init__(self, depth: int, **kwargs):
        """
        Implement the ablation of the method to have fixed depth.
        All probability is concentrated on the last layer.

        :param depth: the fixed depth of the network
        :param kwargs: not used
        """
        super().__init__()
        self.depth = depth

    def to(self, device):
        super().to(device)
        self.device = device

    def get_q_ell_named_parameters(self) -> dict:
        return {}

    def compute_truncation_number(self) -> int:
        return self.depth

    def compute_probability_vector(self) -> torch.Tensor:
        """
        Computes the **renormalized** vector of probabilities on the fly

        :return: a vector of arbitrary length with the probabilities
        """
        depth = self.compute_truncation_number()
        probs = torch.ones(depth + 1, device=self.device)
        probs = probs / probs.sum(0, keepdims=True)
        return probs

    @property
    def mean(self) -> torch.Tensor:
        return -1
