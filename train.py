import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from model import VAE, CrossModalImputationModel, Discriminator
from metric import train_and_evaluate_svm, pca, calculate_r2_per_sample, evaluate_imputation
from torch.utils.data import Dataset, DataLoader


class DataFrameDataset(Dataset):
    def __init__(self, dataframe, label_col=None, transform=None):

        self.dataframe = dataframe
        self.transform = transform

        if label_col is not None:
            if isinstance(label_col, str):
                self.labels = dataframe[label_col].values
                self.features = dataframe.drop(columns=[label_col]).values
            elif isinstance(label_col, int):
                self.labels = dataframe.iloc[:, label_col].values
                self.features = dataframe.drop(dataframe.columns[label_col], axis=1).values
        else:
            self.features = dataframe.values
            self.labels = None

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.features[idx, :].astype(np.float32)

        if self.labels is not None:
            label = self.labels[idx]
            if self.transform:
                sample = self.transform(sample)
            return torch.tensor(sample), torch.tensor(label, dtype=torch.long)
        else:
            if self.transform:
                sample = self.transform(sample)
            return torch.tensor(sample)


class source2targetDataset(Dataset):
    def __init__(self, source_data, target_data, paired=True):
        """
        Args:
            source_data: num_samples, num_feature_source
            target_data: num_samples, num_feature_target
            paired: True means snp_data[i] and gene_data[i] are paired
        """
        self.source_data = torch.FloatTensor(source_data)
        self.target_data = torch.FloatTensor(target_data)
        self.paired = paired

        if not paired:
            self.random_idx = torch.randperm(len(target_data))

    def __len__(self):
        return len(self.source_data)

    def __getitem__(self, idx):
        if self.paired:
            return self.source_data[idx], self.target_data[idx]
        else:
            return self.source_data[idx], self.target_data[self.random_idx[idx]]


def vae_loss(x_recon, x_true, mu, logvar, beta=1.0):

    recon_loss = F.mse_loss(x_recon, x_true, reduction='sum')
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_div


def mmd_loss(z_source, z_target, kernel='rbf', bandwidth=None):

    def gaussian_kernel(x, y, sigma):
        x = x.unsqueeze(1)  # (batch, 1, dim)
        y = y.unsqueeze(0)  # (1, batch, dim)
        return torch.exp(-torch.sum((x - y)**2, dim=-1) / (2 * sigma**2))

    if kernel == 'multi-scale':
        # 多尺度RBF核 (推荐)
        sigmas = [1, 2, 4, 8, 16]
        k_xx = k_yy = k_xy = 0
        for sigma in sigmas:
            k_xx += gaussian_kernel(z_source, z_source, sigma)
            k_yy += gaussian_kernel(z_target, z_target, sigma)
            k_xy += gaussian_kernel(z_source, z_target, sigma)
        k_xx /= len(sigmas)
        k_yy /= len(sigmas)
        k_xy /= len(sigmas)
    else:
        # 单核
        sigma = bandwidth or 1.0
        k_xx = gaussian_kernel(z_source, z_source, sigma)
        k_yy = gaussian_kernel(z_target, z_target, sigma)
        k_xy = gaussian_kernel(z_source, z_target, sigma)

    mmd = k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()
    return torch.clamp(mmd, min=0)


def pretrain(train_data, test_data, num_epoch_pretrain):

    # Hyperparameters
    lr = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize VAE model
    vae = VAE(train_data.shape[1]).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=lr)

    dataset_train = DataFrameDataset(train_data)
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=32,
        shuffle=True,
    )
    dataset_test = DataFrameDataset(test_data)
    dataloader_test = DataLoader(
        dataset_test,
        batch_size=len(test_data),
        shuffle=True,
    )

    for epoch in range(num_epoch_pretrain):
        vae.train()
        epoch_train_loss = 0.0

        for batch_train in dataloader_train:
            batch_train = batch_train.to(device)

            optimizer.zero_grad()
            x_recon, mu, logvar, _ = vae(batch_train)

            loss = vae_loss(x_recon, batch_train, mu, logvar)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / train_data.shape[0]

        # Validation
        vae.eval()
        avg_test_loss = 0.0
        with torch.no_grad():
            for batch_test in dataloader_test:
                batch_test = batch_test.to(device)

                x_recon_test, mu_test, logvar_test, _ = vae(batch_test)
                loss = vae_loss(x_recon_test, batch_test, mu_test, logvar_test)
                avg_test_loss = loss / len(x_recon_test)

        if (epoch + 1) == num_epoch_pretrain:
            print(f'Pretrain loss - [ Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}]')
    return vae


def train(source_vae, target_vae, train_source, train_target, test_source, test_target, num_epochs=1000, lr=0.01):

    lamb = 0.01
    eta = 0.01
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    generator = CrossModalImputationModel(source_vae, target_vae, latent_dim=64)
    discriminator = Discriminator(train_target.shape[1])

    adversarial_loss = nn.BCELoss()
    optimizer_G = torch.optim.Adam(generator.parameters(), lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr)

    train_source = train_source.values
    train_target = train_target.values
    test_source = test_source.values
    test_target = test_target.values

    train_dataset = source2targetDataset(train_source, train_target, paired=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
    )
    train_num_batches = len(train_loader)

    test_dataset = source2targetDataset(test_source, test_target, paired=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_source.shape[0],
        shuffle=True,
    )
    test_num_batches = len(test_loader)

    generator = generator.to(device)
    discriminator = discriminator.to(device)

    for epoch in range(num_epochs):
        avg_train_loss = 0.0
        generator.train()
        for source_batch, target_batch in train_loader:
            source_batch = source_batch.to(device)
            target_batch = target_batch.to(device)

            # Real and Fake labels
            valid = torch.ones(source_batch.size(0), 1).to(device)
            fake = torch.zeros(source_batch.size(0), 1).to(device)

            # ---------------------
            #  Training Generator
            # ---------------------
            optimizer_G.zero_grad()
            gen_target, z_target_pred = generator(source_batch)
            _, _, _, z_source = source_vae(source_batch)
            _, _, _, z_target_true = target_vae(target_batch)

            # 生成器损失：对抗损失 + 重建损失
            g_loss_adv = adversarial_loss(discriminator(gen_target), valid)
            g_loss_rec = torch.abs(gen_target - target_batch).mean(dim=1).mean()
            g_loss_mmd = mmd_loss(z_target_pred, z_target_true)  # MMD损失
            g_loss = eta * g_loss_adv + (1-eta-lamb) * g_loss_rec + lamb * g_loss_mmd

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Training discriminator
            # ---------------------
            optimizer_D.zero_grad()

            real_loss = adversarial_loss(discriminator(target_batch), valid)
            fake_loss = adversarial_loss(discriminator(gen_target.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()
            avg_train_loss += torch.abs(gen_target - target_batch).mean(dim=1).mean()

        # Validation
        if (epoch + 1) % 10 == 0:
            generator.eval()
            avg_test_loss = 0.0
            with torch.no_grad():
                for source_batch, target_batch in test_loader:
                    source_batch = source_batch.to(device)
                    target_batch = target_batch.to(device)
                    outputs, _ = generator(source_batch)
                    avg_test_loss += torch.abs(outputs - target_batch).mean(dim=1).mean()

            print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: { avg_train_loss/train_num_batches:.4f}, Test Loss: {avg_test_loss/test_num_batches:.4f}')

    return generator


def main():

    datasets_name = 'BRCA'
    num_epoch_pretrain = 500
    classification = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if datasets_name == 'ROSMAP':
        source = pd.read_csv('ROSMAP/snp.csv').set_index('Unnamed: 0')
        target = pd.read_csv('ROSMAP/gene_expression.csv').set_index('Unnamed: 0')
        eQTL = pd.read_csv('ROSMAP/eQTL.csv')
        pheno = pd.read_csv('ROSMAP/labels.csv').set_index('sample')[["braaksc_label","ceradsc_label","cogdx_label"]]

        filtered_df = eQTL[eQTL['source'].isin(source.columns)]
        filtered_df = filtered_df[filtered_df['target'].isin(target.columns)]
        source_feature_list = source.columns.values.tolist()
        target_feature_list = target.columns.values.tolist()
        association = filtered_df[['source', 'target']]

        association_matrix = pd.crosstab(
            index=association['source'],
            columns=association['target'],
            dropna=False
        ).clip(upper=1)

        association_matrix = association_matrix.reindex(
            index=source_feature_list,
            columns=target_feature_list,
            fill_value=0
        )
        target = (source.dot(association_matrix) + target).clip(upper=1)

    else:
        source = pd.read_csv('BRCA/DNA methylation.csv', header=None)
        target = pd.read_csv('BRCA/RNA-seq.csv',  header=None)
        pheno = pd.read_csv('BRCA/labels.csv',  header=None)
        # gene_list = gene_data.columns.values.tolist()

    if not source.index.equals(target.index):
        print("Sample mismatch between training and test sets")
        return

    # miss_rate setting
    miss_rate = [0.2, 0.4, 0.6, 0.8]
    for miss_rate in miss_rate:
        print(f'Miss rate is {miss_rate}')
        num_samples = len(target)
        num_train_samples = int((1 - miss_rate) * num_samples)
        train_sample_indices = np.random.choice(target.index, size=num_train_samples, replace=False)

        train_target = target.loc[train_sample_indices]
        train_source = source.loc[train_sample_indices]
        test_target = target.drop(train_sample_indices)
        test_source = source.drop(train_sample_indices)
        print(f'Training-set sample size: {len(train_target)}, Test-set sample size: {len(test_target)}')

        print('Pretraining the VAE for source Data...')
        source_vae = pretrain(train_source, test_source, num_epoch_pretrain)
        print('Pretraining the VAE for target Data...')
        target_vae = pretrain(train_target, test_target, num_epoch_pretrain)
        print('Finished Pretraining!')

        print('Training the imputation model...')
        imputation_model = train(source_vae, target_vae, train_source, train_target, test_source, test_target)
        print('Train Finished, start calculating the evaluation metrics!')

        imputation_model.eval()
        source_tensor = torch.tensor(source.values, dtype=torch.float32).to(device)
        target_tensor = torch.tensor(target.values, dtype=torch.float32).to(device)
        target_imputation, _ = imputation_model(source_tensor)
        calculate_r2_per_sample(target_imputation, target_tensor, miss_rate)

        if classification:
            if datasets_name == 'BRCA':
                print("Classification on imputed data")
                train_and_evaluate_svm(target_imputation, pheno[0])
                print("Classification on real data")
                train_and_evaluate_svm(target_tensor, pheno[0])
                pca(target_imputation, pheno[0], miss_rate, datasets_name)

            if datasets_name == 'ROSMAP':
                print("Classification on imputed data")
                train_and_evaluate_svm(target_imputation, pheno['braaksc_label'])
                print("Classification on real data")
                train_and_evaluate_svm(target_tensor, pheno['braaksc_label'])
                pca(target_imputation, pheno["braaksc_label"], miss_rate, datasets_name)

        evaluate_imputation(target_tensor, target_imputation)

if __name__ == "__main__":
    main()


