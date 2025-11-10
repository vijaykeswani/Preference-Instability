
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
from stability_utils import *
import pingouin as pg
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score

from scipy.stats import kruskal
import shap
from stargazer.stargazer import Stargazer
from sklearn.isotonic import spearmanr

import choix
import random
import warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import gymnasium
from gymnasium import spaces


feats = ['dep', 'alco', 'crim', 'life',
           'years_waiting', 'work_hours', 'obesity', 'reject_chance']

diff_cols = ['dep_diff', 'alco_diff', 'crim_diff', 'life_diff',
            'years_waiting_diff', 'work_hours_diff', 'obesity_diff', 'reject_chance_diff']
l_cols = ['l_' + f for f in feats]
r_cols = ['r_' + f for f in feats]

def get_clf_model(df, reps=5, model_name="lr"):
    x = df[diff_cols]
    y = [int(p) for p in df['chosen']]

    accs, aucs, coefs, shap_coefs = [], [], [], []
    for _ in range(reps):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x_train)

        if model_name == "lr":
            model = LogisticRegression(max_iter = 100)
        elif model_name == "rf":
            model = RandomForestClassifier(n_estimators=100)
        model.fit(x_scaled, y_train)
        acc = model.score(scaler.transform(x_test), y_test)
        y_prob = model.predict_proba(scaler.transform(x_test))[:, 1]
        auc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.5
        aucs.append(auc)
        accs.append(acc)

        if model_name == "lr":
            coefs.append(model.coef_[0])
        else:  
            coefs.append(model.feature_importances_)

        background = shap.maskers.Independent(x_scaled, max_samples=100)
        # X100 = shap.utils.sample(X, 100)
            
        explainer = shap.Explainer(lambda x:model.predict_proba(x)[:, 1], background)
        shap_values = explainer(x_scaled)        
        shap_mean = list(np.abs(shap_values.values).mean(axis=0))

        shap_coefs.append(shap_mean)


    return coefs, shap_coefs, scaler, np.mean(accs), np.mean(aucs)




def get_clf_model_performance(df, model_name="lr", cv=5):

    feat_cols = ["l_"+f for f in feats] + ["r_"+f for f in feats]
    if model_name == "mlp":
        clf = MLPClassifier()
    elif model_name =="lr":
        clf = LogisticRegression()
    scores = cross_val_score(clf, df[feat_cols], df["chosen"], cv=cv)
    clf.fit(df[feat_cols], df["chosen"])

    return np.mean(scores), clf


### New Bradley-Terry model performance function - based on DPO
def get_BT_dpo_model_performance(df, cv=5):

    scores = []
    for _ in range(cv):
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        train_dataset = [(np.array(list(row[l_cols].values)), np.array(list(row[r_cols].values)), row['chosen']) for _, row in train_df.iterrows()]
        test_dataset = [(np.array(list(row[l_cols].values)), np.array(list(row[r_cols].values)), row['chosen']) for _, row in test_df.iterrows()]

        model = PolicyModel(len(feats))
        train_dpo_policy(model, train_dataset, epochs=20)

        acc = []
        for lP, rP, chosen in test_dataset:
            # lfeat = np.array(lP) - np.array(rP)
            score_a = policy_score(model, lP)
            score_b = policy_score(model, rP)
            model_pref = 1 if score_a > score_b else 0

            # print (chosen, model_pref, score_a, score_b)
            acc.append(model_pref == chosen)
        acc = np.mean(acc)
        scores.append(acc)
    return np.mean(scores), model



class PairwisePreferenceDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs  # List of tuples: (a, b, label), where label=1 if a is preferred

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        a, b, label = self.pairs[idx]
        return torch.tensor(a, dtype=torch.float32), torch.tensor(b, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

class RewardModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

def train_reward_model(dataset, input_dim, epochs=10, lr=1e-3):
    model = RewardModel(input_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0
        for a, b, label in loader:
            ra = model(a.to(torch.float32))
            rb = model(b.to(torch.float32))
            prob = torch.sigmoid(ra - rb)
            loss = -label * torch.log(prob + 1e-8) - (1 - label) * torch.log(1 - prob + 1e-8)
            loss = loss.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
    return model

class FeatureVectorEnv(gymnasium.Env):
    def __init__(self, reward_model, d, seed=None):
        super().__init__()
        self.d = d
        self.reward_model = reward_model.eval()
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(d,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(d,), dtype=np.float32)
        self.state = np.random.uniform(-1, 1, size=(d,))
        if seed is not None:
            np.random.seed(seed)

    def step(self, action):
        self.state = np.clip(self.state + action, -1.0, 1.0)
        state_tensor = torch.tensor(self.state, dtype=torch.float32)
        with torch.no_grad():
            reward = self.reward_model(state_tensor).item()
        terminated = False  # Set termination criteria if needed
        truncated = False   # Set truncation criteria if needed
        info = {}           # Additional information can be added here if needed
        return self.state.copy(), reward, terminated, truncated, info

    def reset(self, seed=None):
        self.state = np.random.uniform(-1, 1, size=(self.d,)).astype(np.float32)
        info = {}  # Additional information can be added here if needed
        return self.state.copy(), info

class PolicyModel(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)  # score/logit for input

def dpo_loss(policy, batch, beta=1.0):
    a, b, r = batch
    la = policy(a.to(torch.float32))
    lb = policy(b.to(torch.float32))
    logits = beta * (la - lb)
    return nn.functional.binary_cross_entropy_with_logits(logits, r.float())
    
def train_dpo_policy(policy, dataset, epochs=10, batch_size=16):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    for epoch in range(epochs):
        total_loss = 0
        for a,b,r in loader:
            optimizer.zero_grad()
            loss = dpo_loss(policy, (a,b,r))
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(a)
        # print(f"DPO Policy Epoch {epoch+1}, Loss: {total_loss/len(dataset):.4f}")


def policy_score(policy, obs):
    """Get policy's score or action preference. For simplicity, use value estimate or reward model."""
    if hasattr(policy, "predict_values"):
        return policy.predict_values(torch.tensor(obs, dtype=torch.float32).unsqueeze(0)).item()
    else:
        return policy(torch.tensor(obs, dtype=torch.float32).unsqueeze(0)).item()


def ai_performance_over_time(df, user, results, model_name="clf-mlp", gpt_predictions=None):
    df_user = df[df.tid == user]
    df_user_scaled = df_user.copy()

    dataset_by_session, timestamped_dataset_by_session = {}, {}
    for sess in sorted(df_user_scaled.session_number.unique()):
        df_session = df_user_scaled[df_user_scaled.session_number == sess]
        if sess > 5:
            continue

        dataset, timestamped_dataset = [], []
        for row in df_session.iterrows():
            lP = np.array(list(row[1][l_cols].values))
            rP = np.array(list(row[1][r_cols].values))
            chosen = row[1]['chosen']
            timestamp = (row[1]['session_number']-1)*60 + row[1]['query_num']  # Example timestamp, can be adjusted
            timestamp_dt = row[1]['created_at']
            reaction_time = row[1]['reaction_time']
            # print (timestamp, timestamp_dt)
            pairid = row[1]['pairid']
            # if 'G' in pairid or 'A' in pairid:
                # continue

            dataset.append((lP, rP, chosen))
            timestamped_dataset.append((lP, rP, chosen, timestamp, timestamp_dt, reaction_time))

        dataset_by_session[sess] = list(dataset)
        timestamped_dataset_by_session[sess] = list(timestamped_dataset)
    
    for split_session in (range(1, 5)):

        train_dataset, test_dataset, test_timestamped_dataset = [], [], []
        train_timestamp = None
        for i in dataset_by_session.keys():
            if i == split_session:
                train_dataset += dataset_by_session[i]
                ts = [x[4] for x in timestamped_dataset_by_session[i]]
                train_timestamp = max(ts) 
            if i > split_session:
                test_dataset += dataset_by_session[i]
                test_timestamped_dataset += timestamped_dataset_by_session[i]

        if len(train_dataset) == 0 or len(test_dataset) == 0 or len(test_timestamped_dataset) == 0:
            print(f"Skipping user {user}, split_session {split_session} due to insufficient data.")
            continue

        if model_name == "clf-mlp":
            model = MLPClassifier(hidden_layer_sizes=(32, 8))
            # model = LogisticRegression()
            train_feats = [np.array(lP)-np.array(rP) for (lP, rP, chosen) in train_dataset]
            train_labels = [chosen for (lP, rP, chosen) in train_dataset]
            model.fit(train_feats, train_labels)

        elif model_name == "bt-dpo":
            model = PolicyModel(len(feats))
            train_dpo_policy(model, train_dataset, epochs=20)

        elif model_name == "gpt":
            model = None
        
        dpo_disag = disagreement_over_time(user, model, test_timestamped_dataset, model_name, train_timestamp=train_timestamp, gpt_predictions=gpt_predictions)
        if len(dpo_disag) == 0:
            print(f"No data for user {user} in split_session {split_session}.")
            continue

        ex = np.array([[user, split_session]]*len(dpo_disag))
        dpo_disag = np.append(ex, dpo_disag, axis=1)

        results += list(dpo_disag)
    return results

def disagreement(user, dataset_with_timestamps, model, train_timestamp, model_name, gpt_predictions=None):
    results = []
    for a, b, r, qn, ts, rt in dataset_with_timestamps:
        human_pref = r
        if model_name == "clf-mlp":
            model_pref = model.predict([np.array(a)-np.array(b)])[0]

        elif model_name == "bt-dpo":
            score_a = policy_score(model, a)
            score_b = policy_score(model, b)
            model_pref = 1 if score_a > score_b else 0

        elif model_name == "gpt":
            try:
                model_pref = gpt_predictions[gpt_predictions.tid == user][gpt_predictions.query_num_ts == qn]['GPT2_Pred'].values[0]
                label_pref = gpt_predictions[gpt_predictions.tid == user][gpt_predictions.query_num_ts == qn]['label'].values[0]
                # print (f"{user} found query {qn}", r, label_pref, model_pref)
            except:
                print(f"GPT2 prediction not found for user {user}, query {qn}.")
                continue
        
        time_since = pd.to_datetime(ts) - pd.to_datetime(train_timestamp)
        results.append([qn, time_since, rt, model_pref, human_pref, model_pref != human_pref])

        
    return results

def disagreement_over_time(user, model, dataset_with_timestamps, model_name, train_timestamp=None, gpt_predictions=None):
    
    dataset_with_timestamps.sort(key=lambda x: x[3])

    disag_results = disagreement(user, dataset_with_timestamps, model, train_timestamp, model_name, gpt_predictions=gpt_predictions)
    return np.array(disag_results)



     
### OLD Bradley-Terry model performance function
def get_BT_model_performance(df, model_name="lr", reps=5):

    df["LP"] = df.apply(lambda row: "_".join([str(row["l_"+f]) for f in feats]), axis=1)
    df["RP"] = df.apply(lambda row: "_".join([str(row["r_"+f]) for f in feats]), axis=1)
    pairs = df[["LP", "RP", "chosen"]].to_numpy()

    pairs_bt, profiles = [], []
    for lp, rp, chosen in pairs:
        if chosen == 1:
            pairs_bt.append([lp, rp])
        else:
            pairs_bt.append([rp, lp])
        profiles.append(lp)
        profiles.append(rp)

    profiles = np.unique(profiles)
    inv_profile_map = {p: i for i, p in enumerate(profiles)}
    pairs_num = []
    for lp, rp in pairs_bt:
        pairs_num.append([inv_profile_map[lp], inv_profile_map[rp]])

    accs = []
    for _ in range(reps):
        random.shuffle(pairs_num)
        N = int(0.7*len(pairs))
        train_pairs = pairs_num[:N]
        test_pairs = pairs_num[N:]

        BT_model_scores = choix.opt_pairwise(len(profiles), train_pairs)

        prof_feats, scores = [], []
        for i in range(len(profiles)):
            score = BT_model_scores[i]
            profile = profiles[i]
            feat = profile.split("_")
            feat = [int(f) for f in feat]

            scores.append(score)
            prof_feats.append(feat)

            # print (score, feat)

        if model_name == "mlp":
            model = MLPRegressor(random_state=1, max_iter=2000, tol=0.1)
            model.fit(prof_feats, scores)

        elif model_name == "lr":
            model = LinearRegression()
            model.fit(prof_feats, scores)

        preds = []
        for lp, rp in test_pairs:
            lfeat = profiles[lp]
            lfeat = [int(f) for f in lfeat.split("_")]
            rfeat = profiles[rp]
            rfeat = [int(f) for f in rfeat.split("_")]
            scores = model.predict([lfeat, rfeat])

            pred = int(scores[0] > scores[1])
            preds.append(pred)

        accs.append(np.mean(preds))

    return accs, model


