import math
import torch.nn as nn
from torch.distributions import MultivariateNormal, Categorical
import torch


def gaussian_probability(mu, sigma, rho, data):
    mean_y, mean_x = torch.chunk(mu, 2, dim=-1)
    std_y, std_x = torch.chunk(sigma, 2, dim=-1)
    y, x = torch.chunk(data, 2, dim=2)
    dx = x - mean_x
    dy = y - mean_y
    std_xy = std_x * std_y
    z = (dx * dx) / (std_x * std_x) + (dy * dy) / (std_y * std_y) - (2 * rho * dx * dy) / std_xy
    training_stablizer = 2
    norm = 1 / (training_stablizer * math.pi * std_x * std_y * torch.sqrt(1 - rho * rho))
    p = norm * torch.exp(-z / (1 - rho * rho) * 0.5)
    return p


def mixture_probability(pi, mu, sigma, rho, data):
    pi = pi.unsqueeze(-1)
    prob = pi * gaussian_probability(mu, sigma, rho, data)
    prob = torch.sum(prob, dim=2)
    return prob


def sample_mdn_simp(pi, mu, sigma, rho, sample_num=1):
    pi, mu, sigma, rho = pi.float().cpu(), mu.float().cpu(), sigma.float().cpu(), rho.float().cpu()
    batch_size = pi.size(0)
    max_length = pi.size(1)
    cat = Categorical(pi)
    pis = cat.sample()
    samples_seq = torch.zeros(batch_size, sample_num, max_length, 2)
    samples_neg_log_probs = torch.zeros(batch_size, sample_num, max_length)
    for index in range(batch_size):
        for num in range(sample_num):
            for t in range(max_length):
                idx = pis[index, t]
                loc = mu[index, t, idx]
                std = sigma[index, t, idx]
                std_y, std_x = std[0].item(), std[1].item()
                r = rho[index, t, idx].item()
                cov_mat = torch.tensor([[std_y * std_y, std_y * std_x * r], [std_y * std_x * r, std_x * std_x]])

                MN = MultivariateNormal(loc, covariance_matrix=cov_mat)
                fixation = MN.sample()
                neg_log_probs = -MN.log_prob(fixation)  # ?
                samples_neg_log_probs[index, num, t] = neg_log_probs

                samples_seq[index, num, t, 0] = fixation[0]
                samples_seq[index, num, t, 1] = fixation[1]

    return samples_seq, samples_neg_log_probs


def sample_mdn_one(pi, mu, sigma, rho):
    pi, mu, sigma, rho = pi.float().cpu(), mu.float().cpu(), sigma.float().cpu(), rho.float().cpu()
    max_length = pi.size(0)
    cat = Categorical(pi)
    pis = cat.sample()
    samples_seq = torch.zeros(max_length, 2)
    for t in range(max_length):
        idx = pis[t]
        loc = mu[t, idx]
        std = sigma[t, idx]
        std_y, std_x = std[0].item(), std[1].item()
        r = rho[t, idx].item()
        cov_mat = torch.tensor([[std_y * std_y, std_y * std_x * r], [std_y * std_x * r, std_x * std_x]])

        MN = MultivariateNormal(loc, covariance_matrix=cov_mat)
        fixation = MN.sample()
        samples_seq[t, 0] = fixation[0]
        samples_seq[t, 1] = fixation[1]

    return samples_seq


def sample_mdn(pi, mu, sigma, rho):
    pi, mu, sigma, rho = pi.float().cpu(), mu.float().cpu(), sigma.float().cpu(), rho.float().cpu()
    cat = Categorical(pi)
    pis = cat.sample()
    samples_fixation = torch.zeros(2)

    idx = pis
    loc = mu[idx]
    std = sigma[idx]
    std_y, std_x = std[0].item(), std[1].item()
    r = rho[idx].item()
    cov_mat = torch.tensor([[std_y * std_y, std_y * std_x * r], [std_y * std_x * r, std_x * std_x]])

    MN = MultivariateNormal(loc, covariance_matrix=cov_mat)
    fixation = MN.sample()
    samples_fixation[0] = fixation[0]
    samples_fixation[1] = fixation[1]

    return samples_fixation


class MDN(nn.Module):
    def __init__(self, input_dim, MDN_hidden_num, output_dim, num_gaussians,):
        super(MDN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_gaussians = num_gaussians     # 混合网络个数

        self.pi = nn.Sequential(
            nn.Linear(self.input_dim, MDN_hidden_num),
            nn.ReLU(),
            nn.Linear(MDN_hidden_num, self.num_gaussians),
            nn.Softmax(dim=-1)
        )
        self.mu = nn.Sequential(
            nn.Linear(self.input_dim, MDN_hidden_num),
            nn.ReLU(),
            nn.Linear(MDN_hidden_num, self.output_dim * self.num_gaussians)
        )
        self.std = nn.Sequential(
            nn.Linear(self.input_dim, MDN_hidden_num),
            nn.ReLU(),
            nn.Linear(MDN_hidden_num, self.output_dim * self.num_gaussians)
        )
        self.rho = nn.Sequential(
            nn.Linear(self.input_dim, MDN_hidden_num),
            nn.ReLU(),
            nn.Linear(MDN_hidden_num, self.num_gaussians)
        )
        self.mu[-1].bias.data.copy_(torch.rand_like(self.mu[-1].bias))

    def forward(self, x):
        pi = self.pi(x)
        mu = self.mu(x)
        sigma = torch.exp(self.std(x))
        # print(sigma.mean(0))
        # sigma = torch.clamp(sigma, 0.06, 10)
        rho = torch.tanh((self.rho(x)))

        # rho = torch.clamp(self.rho(x), -0.25, 0.25)
        mu = mu.reshape(-1, mu.size(1), self.num_gaussians, self.output_dim)
        sigma = sigma.reshape(-1, sigma.size(1), self.num_gaussians, self.output_dim)
        rho = rho.reshape(-1, rho.size(1), self.num_gaussians, 1)

        return pi, mu, sigma, rho

    def mixture_probability_map(self, pi, mu, sigma, rho, hw_t):
        pi = pi.unsqueeze(-1)
        prob = pi * gaussian_probability(mu, sigma, rho, hw_t.unsqueeze(0))
        prob = torch.sum(prob, dim=2)
        return prob

    def sample_mdn(self, pi, mu, sigma, rho, sample_num=10):
        max_length = pi.size(1)
        pi, mu, sigma, rho = pi.float(), mu.float(), sigma.float(), rho.float()
        cat = Categorical(pi)
        pis = cat.sample()
        samples = list()
        samples_neg_log_probs = list()
        for num in range(sample_num):
            mask_index = torch.arange(5).unsqueeze(0).unsqueeze(0).expand(self.cfg.train_batch_size, max_length, self.cfg.num_gauss).to(self.cfg.device) \
                         == pis.unsqueeze(-1).expand(self.cfg.train_batch_size, max_length, self.cfg.num_gauss)

            mu_sample = mu[mask_index].reshape(self.cfg.train_batch_size, max_length, 2)  # ??
            sigma_sample = sigma[mask_index].reshape(self.cfg.train_batch_size, max_length, 2)
            rho_sample = rho[mask_index].reshape(self.cfg.train_batch_size, max_length)

            loc = mu_sample
            std_y = sigma_sample[:, :, 0]
            std_x = sigma_sample[:, :, 1]
            r = rho_sample
            cov_mat = torch.zeros(self.cfg.train_batch_size, max_length, 2, 2).to(self.cfg.device)
            cov_mat[:, :, 0, 0] = std_y * std_y
            cov_mat[:, :, 0, 1] = std_y * std_x * r
            cov_mat[:, :, 1, 0] = std_y * std_x * r
            cov_mat[:, :, 1, 1] = std_x * std_x

            MN = MultivariateNormal(loc, covariance_matrix=cov_mat)
            fixations = MN.sample()
            neg_log_probs = -MN.log_prob(fixations)

            samples.append(fixations.unsqueeze(0))
            samples_neg_log_probs.append(neg_log_probs.unsqueeze(0))
        samples = torch.cat(samples, dim=0).transpose(0, 1)
        samples_neg_log_probs = torch.cat(samples_neg_log_probs, dim=0).transpose(0, 1)
        return samples, samples_neg_log_probs

