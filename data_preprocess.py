# ROSMAP data preprocess
import pandas as pd
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import MinMaxScaler


def snp_load():
    all_sample = pd.read_csv("all_sample.csv")["gwas_id"].values.tolist()
    all_snp = ["sample"]

    all_data = []
    for rank in range(1, 22):
        data = pd.read_csv('original_SNP/chr'+str(rank)+'_filter_snp.csv').values.tolist()
        for item in data:
            specific_data = []
            sample_data = item[1].split(" ")[2:-1]
            if len(sample_data) == 1708:
                all_snp.append(item[0])
                series = pd.to_numeric(pd.Series(sample_data), errors="coerce")
                float_list = series.dropna().tolist()
                specific_data.extend(float_list)
            all_data.append(specific_data)
    df = pd.DataFrame(all_data).T
    df.insert(loc=0, column='New_Column', value=all_sample)
    df.columns = all_snp
    return df


def gene_load():
    gene_data = pd.read_csv("gene_expression.csv").iloc[:, 2:]
    sample = gene_data.columns.values.tolist()[1:]
    sample = [s.replace("_", "") for s in sample]
    column_name = ["sample"]
    column_name.extend(gene_data["Symbol"].values.tolist())
    gene_data = gene_data.iloc[:, 1:].values.tolist()

    df = pd.DataFrame(gene_data).T
    df.insert(loc=0, column='New_Column', value=sample)
    df.columns = column_name
    return df


def clinic_load():
    clinic_data = pd.read_csv("ROSMAP_clinical.csv")
    clinic_data['sample'] = clinic_data['Study'] + clinic_data['projid'].astype(str)
    clinic = clinic_data[["sample","braaksc","ceradsc","cogdx"]]
    cols_to_check = ['braaksc', 'ceradsc', 'cogdx']
    clinic = clinic.dropna(subset=cols_to_check, how='all')
    return clinic


def data_get():
    snp_data = snp_load().set_index("sample")
    snp_sample = snp_data.index.values.tolist()
    clinic_data = clinic_load().set_index("sample")
    clinic_sample = clinic_data.index.values.tolist()
    gene_data = gene_load().set_index("sample")
    gene_sample = gene_data.index.values.tolist()

    intersection_snp_clinic = set(snp_sample).intersection(set(clinic_sample))
    intersection_all = intersection_snp_clinic.intersection(set(gene_sample))

    # samples who only have snp_dosage data need to be imputed
    sample_only_snp = list(set(intersection_snp_clinic) - set(gene_sample))
    filtered_snp_only = snp_data.loc[snp_data.index.isin(sample_only_snp)].sort_index()
    filtered_clinic_only = clinic_data.loc[clinic_data.index.isin(sample_only_snp)].sort_index()

    # samples who have snp_dosage data and gene-expression data
    filtered_snp_all = snp_data.loc[snp_data.index.isin(intersection_all)].sort_index()
    filtered_gene_all = gene_data.loc[gene_data.index.isin(intersection_all)].sort_index()
    filtered_clinic_all = clinic_data.loc[clinic_data.index.isin(intersection_all)].sort_index()

    # 1222 samples = 398 samples with both data + 824 samples with only snp
    return filtered_snp_all, filtered_gene_all, clinic_label(filtered_clinic_all), filtered_snp_only, filtered_clinic_only


def clinic_label(clinic_data):
    phenotype_label = ["braaksc", "ceradsc", "cogdx"]
    for element in phenotype_label:
            label_list = []
            if element == "braaksc":
                pheno_list = clinic_data["braaksc"].values.tolist()
                for score in pheno_list:
                    if score > 3:
                        label_list.append(1)
                    else:
                        label_list.append(0)
                clinic_data["braaksc_label"] = label_list
            elif element == "ceradsc":
                pheno_list = clinic_data["ceradsc"].values.tolist()
                for score in pheno_list:
                    if score > 2:
                        label_list.append(0)
                    elif score == 2:
                        label_list.append(1)
                    else:
                        label_list.append(2)
                clinic_data["ceradsc_label"] = label_list
            else:
                pheno_list = clinic_data["cogdx"].values.tolist()
                for score in pheno_list:
                    if score > 3:
                        label_list.append(2)
                    elif score < 2:
                        label_list.append(0)
                    else:
                        label_list.append(1)
                clinic_data["cogdx_label"] = label_list
    return clinic_data


def feature_select(snp, gene, clinic):
    snp_data = snp.T
    gene_data = gene.T

    eqtl = pd.read_csv("eQTL.csv")
    eqtl = eqtl[['source', 'target']]
    # print('intermediate file:2', eqtl.shape) -> intermediate file:2 (24676, 2)

    # # Read GRN data
    grn = pd.read_csv("GRN_file.csv")
    grn = grn[['Transcription_Factor', 'Target_Gene']]
    grn.columns = ['source', 'target']
    # print('intermediate file:1', grn.shape) -> intermediate file:1 (4664714, 2)

    snp_eqtl_list = eqtl["source"].values.tolist()
    snp_eQTL = set(snp_eqtl_list)
    snp_name = set(snp_data.index.values.tolist())
    common_snp = list(snp_name.intersection(snp_eQTL))
    snp_data = snp_data.loc[common_snp]
    # print(snp_data.shape) -> (13860, 398)

    gene_grn_list = grn['source'].values.tolist()
    gene_grn_list.extend(grn['target'].values.tolist())
    gene_grn = set(gene_grn_list)
    gex_name = set(gene_data.index.values.tolist())
    common_gene = list(gex_name.intersection(gene_grn))
    gex_data = gene_data.loc[common_gene]
    # print(gex.shape) -> (18407, 398)

    gex = variance_filter(gex_data)
    snp = variance_filter(snp_data)
    label = clinic["braaksc_label"]

    return preprocess_ANOVA(gex, snp, label)


def variance_filter(data):
        data = data.T
        variances = data.var()
        high_variance_features = variances[variances > 0.1].index
        filtered_df = data[high_variance_features]
        return filtered_df


def preprocess_ANOVA(gene_feature=None, snp_feature=None, labels=None):

        samples = snp_feature.index.values.tolist()
        labels = labels.to_numpy()
        # 1. calculate ANOVA F-value
        SNP_F_values, SNP_p_values = f_classif(snp_feature, labels)
        gene_F_values, gene_p_values = f_classif(gene_feature, labels)

        # 2. select feature
        SNP_selected_features = snp_feature.loc[:, SNP_p_values < 0.05]
        SNP_feature_list = SNP_selected_features.columns.tolist()

        gene_selected_features = gene_feature.loc[:, gene_p_values < 0.1]
        gene_feature_list = gene_selected_features.columns.tolist()
        # print(f"Selected gene_features shape: {gene_selected_features.shape}")

        scaler1 = MinMaxScaler()
        SNP_normalized_features = scaler1.fit_transform(SNP_selected_features)
        scaler2 = MinMaxScaler()
        gene_normalized_features = scaler2.fit_transform(gene_selected_features)

        SNP_normalized_features = pd.DataFrame(SNP_normalized_features, columns=SNP_feature_list, index=samples)
        gene_normalized_features = pd.DataFrame(gene_normalized_features, columns= gene_feature_list, index=samples)

        return SNP_normalized_features, gene_normalized_features


if __name__ == "__main__":
    snp1, gene1, clinic1_labels, snp2, clinic2 = data_get()
    omic1, omic2 = feature_select(snp1, gene1, clinic1_labels)

    omic1.to_csv('precessed_data/snp.csv')
    omic2.to_csv('precessed_data/gene_expression.csv')
    clinic1_labels.to_csv('precessed_data/labels.csv')










