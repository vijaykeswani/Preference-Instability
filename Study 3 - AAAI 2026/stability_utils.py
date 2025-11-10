import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from repeated_scenarios import *
import seaborn as sns
import scipy
from tqdm.notebook import tqdm
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import pearsonr
import statsmodels.api as sm
import statsmodels.formula.api as smf
import itertools



def get_stability(responses, method="stability"):
    if method=="stability":
        stab = np.mean(responses)
        stab = np.max([stab, 1-stab])
    if method=="std":
        stab = np.std(responses)
    if method=="consistency_continuous":
        stab = np.mean([1 if responses[i]==responses[i+1] else 0 for i in range(len(responses)-1)])
    if method=="consistency_sessions":
        resp_sessions = [np.mean([responses[i*2], responses[i*2+1]]) for i in range(int(len(responses)/2))]
        stab = np.mean([1 if resp_sessions[i]==resp_sessions[i+1] else 0 for i in range(len(resp_sessions)-1)])

    return stab


def get_all_repeated_responses(df, scenarios, limit=6):
    all_responses = {}
    for id in tqdm(df['tid'].unique()):
        responses = {sc: [] for sc in scenarios}    
        df_user = df[df.tid == id]
        sessions = sorted(df_user['session_number'].unique() )
        for session in sessions:
            df_user_session = df_user[df_user.session_number == session]
            for pairid in scenarios:
                    resp1 = list(df_user_session[df_user_session.pairid == pairid]['chosen'])
                    time1 = list(df_user_session[df_user_session.pairid == pairid]['created_at'])
                    # responses[pairid] += (resp)

                    resp2 = list(df_user_session[df_user_session.pairid == pairid+"_r"]['chosen'])
                    resp2 = [1-r for r in resp2]
                    time2 = list(df_user_session[df_user_session.pairid == pairid+"_r"]['created_at'])

                    if len(resp1) == 0 or len(resp2) == 0:
                        resp = resp2 + resp1
                    else:                    
                        if time1[0] > time2[0]:
                            resp = resp2 + resp1
                        else:
                            resp = resp1 + resp2

                    responses[pairid] += (resp)

        if len(responses[scenarios[0]]) < limit:
            # un += 1
            continue


        all_responses[int(id)] = dict(responses)
    return all_responses


def plot_stability_correlation(df, scenarios, xl="difficulty_score", xlabel="Priority score difference", fname=None):

    plt.figure(figsize=(7, 6))

    labs = scenarios

    # c = ["green", "steelblue", "purple", "green", "steelblue", "purple"]
    c = plt.cm.viridis([0, 0, 0.5, 0.5, 0.8, 0.8])
    pairids = sorted(df.pairid.unique())
    for i, pairid in enumerate(pairids):
        
        df_sub = df[df["pairid"] == pairid]
        df_sub = df_sub[df_sub[xl].notna()]
        plt.subplot(3,2,i+1)
        sns.regplot(data=df_sub, x=xl, y='stability', color=c[i], marker='o', scatter_kws={'s': 10, 'alpha':0.5})
        print (pairid, scipy.stats.spearmanr(df_sub[xl], df_sub["stability"]))

        corr = scipy.stats.spearmanr(df_sub[xl], df_sub["stability"]).statistic
        corr = np.round(corr, 2)
        pval = scipy.stats.spearmanr(df_sub[xl], df_sub["stability"]).pvalue
        pind = "**" if pval<0.05 else ""
        plt.xlabel(xlabel, fontsize=9)
        plt.ylabel("RS$(\cdot, \cdot)$", fontsize=10)
        plt.ylim([0.41, 1.09])
        plt.tick_params(axis='both', which='major', labelsize=8)
        
        title = plt.title("Scenario " + labs[i] + "; Spearman $\\rho$: "+ str(corr)+pind, backgroundcolor='white', color='black', fontsize=10)
        title._bbox_patch._mutation_aspect = 0.13
        
    plt.subplots_adjust(hspace=0.65, wspace=0.3)
    if fname is not None:
        plt.savefig(fname, bbox_inches='tight')
    plt.show()
