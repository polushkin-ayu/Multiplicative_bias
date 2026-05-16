import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import logit
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score, r2_score
from scipy.optimize import curve_fit

def generate_bucket(df: pd.DataFrame, field_nm:str, num_buckets:int, nans_filler=0, labels=False, method='other') -> pd.DataFrame:
        '''
        Функция генерирует num_buckets для поля field_nm в датафрейме df
        '''
        max_num_buckets = min(df[field_nm].fillna(nans_filler).nunique(),num_buckets)
        if method == 'q':
            df[field_nm + '_bucket'] = pd.qcut(df[field_nm].fillna(nans_filler),
                                           q=max_num_buckets, labels=labels, duplicates='drop')
        else:
            df[field_nm + '_bucket'] = pd.cut(df[field_nm].fillna(nans_filler),
                                           bins=max_num_buckets, labels=labels, duplicates='drop')

        return df

def plot_line(df, feature_nm, target_nm, n_buckets = 10):
        df_ = generate_bucket(df, feature_nm, n_buckets)

        evaluation_model = LinearRegression()
        evaluation_model.fit(np.array(df_[feature_nm]).reshape(-1,1), df_[target_nm])
        evaluation_r2 = r2_score(df_[target_nm], evaluation_model.predict(np.array(df_[feature_nm]).reshape(-1,1)))

        df_grouped = df_.groupby(feature_nm + '_bucket').agg(
                mean_target = (target_nm, 'mean'),
                mean_feature = (feature_nm, 'mean'),
                std_target = (target_nm, 'std')
        )

        plt.figure(figsize=(10, 5))

        plt.title(f'Качество {feature_nm} на {target_nm}, $R^2 = {evaluation_r2:.4f}$')

        regression_line = evaluation_model.predict(np.array(df_grouped.mean_feature).reshape(-1,1))
        x = df_grouped.mean_feature
        y = df_grouped.mean_target
        err = df_grouped.std_target

        plt.errorbar(x, y, err, fmt='o', label='mean_target')
        plt.plot(x, regression_line, label='regression_line', color = 'red')
        plt.legend()
        plt.show()

def woe_line(df, score, target, n_buckets=5):

        lr = LogisticRegression()

        df_ = df.copy()
        df_[score] = df_[score]
        df_ = generate_bucket(df, score, n_buckets)

        mean_by_sample = df_[target].mean()
        # n_obs_by_sample = df_[target].count()

        woe = np.log(mean_by_sample / (1 - mean_by_sample))

        df_grouped = df_.groupby(score + '_bucket')\
            .agg(mean_target = (target, 'mean'),
                 n_target = (target, 'sum'),
                 n_obs = (target, 'count'),
                 mean_score = (score, 'mean')
                 )\
                 .reset_index()

        df_grouped['woe'] = np.log(df_grouped['mean_target'] / (1 - df_grouped['mean_target']))
        df_grouped['se_woe'] = 1 / np.sqrt(df_grouped['n_obs'] * df_grouped['mean_target'] * (1 - df_grouped['mean_target']))

        fit_ = lr.fit(np.array(df[score]).reshape(-1, 1), np.array(df[target]))

        y_preds = fit_.predict_proba(np.array(df_grouped.mean_score).reshape(-1, 1))

        r2 = r2_score(df_grouped.woe, logit(y_preds[:, 1]))
        auc = roc_auc_score(np.array(df[target]), np.array(df[score]))

        plt.figure(figsize=(10,5))

        plt.title(f'WoE by bucket for {score}, AUC = {auc:.2f}, $R^2$ = {r2:.2f}')

        plt.errorbar(df_grouped.mean_score, df_grouped.woe, yerr=df_grouped.se_woe,
                     label = 'WoE by bucket',color = 'lightblue',fmt = 'o')

        plt.plot(df_grouped.mean_score, logit(y_preds[:,1]), color = 'red', label = 'Logistic regression')

        plt.xlabel(f'{score} bucket')
        plt.ylabel(f'WoE by {target}')

        plt.legend()

        plt.show()
