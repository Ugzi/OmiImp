import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.LeakyReLU(0.2))
            prev_dim = h_dim

        self.feature_extractor = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)

    def forward(self, x):
        h = self.feature_extractor(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []
        prev_dim = latent_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.LeakyReLU(0.2))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Sigmoid())

        self.decoder = nn.Sequential(*layers)

    def forward(self, z):
        return self.decoder(z)


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256,32], latent_dim=64):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dims, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dims[::-1], input_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar, z


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dims=[32]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Dropout(0.3))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class ZMappingNetwork(nn.Module):

    def __init__(self, latent_dim):
        super(ZMappingNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim)
        )

    def forward(self, z_snp):
        return self.net(z_snp)


class CrossModalImputationModel(nn.Module):

    def __init__(self, snp_vae, gene_vae, latent_dim):
        super(CrossModalImputationModel, self).__init__()
        self.snp_vae = snp_vae
        self.gene_vae = gene_vae
        self.mapping_net = ZMappingNetwork(latent_dim)

        for param in self.snp_vae.parameters():
            param.requires_grad = False

    def forward(self, snp_data):
        _, _, _, z_snp = self.snp_vae(snp_data)
        z_gene_pred = self.mapping_net(z_snp)
        gene_pred = self.gene_vae.decoder(z_gene_pred)
        return gene_pred, z_gene_pred
