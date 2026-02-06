#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pickle
import os
import cmdstanpy

abs_path = os.path.dirname(os.path.abspath(__file__))


# =============================================================================
# Model functions
# =============================================================================
def compile_stan_model(force=False):
    """Autocompile Stan model."""
    source_path = os.path.join(abs_path, 'model.stan')
    target_path = os.path.join(abs_path, 'model.bin')
    exe_path = os.path.join(abs_path, 'model')

    # 检查是否需要重新编译
    need_compile = force
    
    # 如果已经存在编译信息，检查模型是否有变化
    if os.path.exists(target_path) and not force:
        try:
            with open(target_path, 'rb') as f:
                model_info = pickle.load(f)
            
            if isinstance(model_info, dict) and 'model_code' in model_info:
                with open(source_path, 'r') as f:
                    file_content = "".join([line for line in f])
                if file_content != model_info['model_code']:
                    need_compile = True
            else:
                # 旧版本存储格式不兼容，需要重新编译
                need_compile = True
        except:
            # 如果加载失败，强制重新编译
            need_compile = True
    else:
        # 如果文件不存在，需要编译
        need_compile = True
    
    if need_compile:
        print(source_path, "[Compiling]", ["", "[Forced]"][force])
        # 使用cmdstanpy编译模型
        model = cmdstanpy.CmdStanModel(stan_file=source_path, model_name="plant_pol")
        # 保存模型信息用于后续检查
        with open(target_path, 'wb') as f:
            model_info = {
                'model_code': open(source_path, 'r').read(),
                'exe_path': model.exe_file
            }
            pickle.dump(model_info, f)
    else:
        print(target_path, "[Skipping --- already compiled]")


def load_model():
    """Load the model to memory."""
    compile_stan_model()
    
    # 获取编译后的模型路径
    try:
        with open(os.path.join(abs_path, "model.bin"), 'rb') as f:
            model_info = pickle.load(f)
        
        # 检查model_info格式是否正确
        if isinstance(model_info, dict) and 'exe_path' in model_info:
            # 返回cmdstanpy模型对象
            return cmdstanpy.CmdStanModel(exe_file=model_info['exe_path'])
        else:
            # 如果格式不正确，重新编译
            print("Model info format incorrect, recompiling...")
            compile_stan_model(force=True)
            with open(os.path.join(abs_path, "model.bin"), 'rb') as f:
                model_info = pickle.load(f)
            return cmdstanpy.CmdStanModel(exe_file=model_info['exe_path'])
    except:
        # 如果加载失败，重新编译
        print("Failed to load model, recompiling...")
        compile_stan_model(force=True)
        with open(os.path.join(abs_path, "model.bin"), 'rb') as f:
            model_info = pickle.load(f)
        return cmdstanpy.CmdStanModel(exe_file=model_info['exe_path'])


# =============================================================================
# Sampling functions
# =============================================================================
def generate_sample(M, C, model, num_chains=4, warmup=5000, num_samples=500):
    """Run sampling for data matrix M."""
    # 准备数据字典
    data = {
        "n": M.shape[0],
        "M": M,
        "C": C,
    }
    
    # 在cmdstanpy中进行采样
    fit = model.sample(
        data=data,
        chains=num_chains,
        iter_warmup=warmup,
        iter_sampling=num_samples,
        show_progress=True,
        max_treedepth=15
    )
    
    return fit


def save_samples(samples, fpath='samples.bin'):
    """Save samples as binaries, with pickle."""
    with open(fpath, 'wb') as f:
        pickle.dump(samples, f)


def load_samples(fpath='samples.bin'):
    """Load samples from binaries, with pickle."""
    with open(fpath, 'rb') as f:
        return pickle.load(f)


def test_samples(samples, tol=0.1, num_chains=4):
    """Verify that no chain has a markedly lower average log-probability."""
    # 从cmdstanpy中获取对数概率值
    log_probs = samples.stan_variable('lp__')
    
    # 按链分割日志概率
    chain_indices = np.array(samples.chains)
    unique_chains = np.unique(chain_indices)
    
    # 计算每条链的平均对数概率
    log_probs_means = np.array([np.mean(log_probs[chain_indices == i]) for i in unique_chains])
    
    return np.alltrue(log_probs_means - (1 - tol) * max(log_probs_means) > 0)


# =============================================================================
# Inference functions
# =============================================================================
def get_posterior_predictive_matrix(samples):
    """Calculate the posterior predictive matrix."""
    # 从cmdstanpy获取参数
    Q = samples.stan_variable('Q')
    C = samples.stan_variable('C')
    r = samples.stan_variable('r')
    k = samples.stan_variable('k')  # 假设模型中有k变量
    
    num_samples = Q.shape[0]
    ones = np.ones((num_samples, Q.shape[1], Q.shape[2]))
    
    sigma = samples.stan_variable('sigma')
    tau = samples.stan_variable('tau')
    
    # 计算sigma_tau (外积)
    sigma_tau = np.einsum('ki,kj->kij', sigma, tau)
    
    # 计算累积 (posterior predictive)
    accu = (1 - Q) * np.einsum('kij,k->kij', ones, C) * sigma_tau
    accu += Q * np.einsum('kij,k->kij', ones, C * (1 + k * r)) * sigma_tau
    
    return np.mean(accu, axis=0)


def estimate_network(samples):
    """Return the matrix of edge probabilities P(B_ij=1)."""
    Q = samples.stan_variable('Q')
    return np.mean(Q, axis=0)


def get_network_property_distribution(samples, property_func, num_net=10):
    """Return the average posterior value of an arbitrary network property.

    Input
    -----
    samples: CmdStanMCMC object
        The posterior samples.
    property_func: function
        This function should take an incidence matrix as input and return a
        scalar.
    num_net: int
        Number of networks to generate for each parameter samples.
    """
    Q = samples.stan_variable('Q')
    num_samples = Q.shape[0]
    
    values = np.zeros(num_samples * num_net)
    
    for i in range(num_samples):
        for j in range(num_net):
            B = np.random.binomial(n=1, p=Q[i])
            values[i * num_net + j] = property_func(B)
            
    return values