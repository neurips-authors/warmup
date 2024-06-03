import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from scipy.interpolate import griddata


plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"font.size": 28, 'lines.linewidth': 3.0})

def get_rows_where_col_equals(df, col, value):
    return df.loc[df[col] == value].copy()

def get_rows_where_col_in(df, col, values):
    return df.loc[df[col].isin(values)].copy()

def get_rows_where_col_greater(df, col, value):
    return df.loc[df[col] > value].copy()

def get_rows_where_col_less(df, col, value):
    return df.loc[df[col] < value].copy()

out_dims = {'cifar-10': 10, 'cifar-100': 100, 'tiny-imagenet': 200}

parser = argparse.ArgumentParser(description = 'Experiment parameters')
parser.add_argument('--cluster', type = str, default = 'zaratan')
# Dataset parameters
parser.add_argument('--dataset', type = str, default = 'cifar-10')
parser.add_argument('--out_dim', type = int, default = 10)
parser.add_argument('--num_examples', type = int, default = 50000)

# Model parameters
parser.add_argument('--abc', type = str, default = 'sp')
parser.add_argument('--width', type = int, default = 512)
parser.add_argument('--widening_factor', type = int, default = 1)
parser.add_argument('--depth', type = int, default = 4)
parser.add_argument('--bias', type = str, default = 'True') # careful about the usage
parser.add_argument('--act_name', type = str, default = 'relu')
parser.add_argument('--init_seed', type = int, default = 1)
parser.add_argument('--scale', type = float, default = 0.0)
parser.add_argument('--varw', type = float, default = 2.0)
#Optimization parameters
parser.add_argument('--loss_name', type = str, default = 'mse')
parser.add_argument('--augment', type = str, default = 'False')
parser.add_argument('--opt_name', type = str, default = 'base_adam')
parser.add_argument('--sgd_seed', type = int, default = 1)
parser.add_argument('--warmup_steps', type = int, default = 1)
parser.add_argument('--warmup_exponent', type = float, default = 1.0)
parser.add_argument('--decay_exponent', type = float, default = 0.0)
parser.add_argument('--num_steps', type = int, default = 2048)
parser.add_argument('--lr_trgt', type = float, default = 0.001)
parser.add_argument('--lr_step', type = float, default = 1.0)
parser.add_argument('--lr_init', type = float, default = 0.0)
parser.add_argument('--b1', type = float, default = 0.9)
parser.add_argument('--b2', type = float, default = 0.999)
parser.add_argument('--batch_size', type = int, default = 512)
# Sharpness estimation
parser.add_argument('--topk', type = int, default = 1)
parser.add_argument('--sharpness_method', type = str, default = 'lobpcg')
parser.add_argument('--measure_batches', type = int, default = 10)
parser.add_argument('--tol', type = float, default = 1e-08)
parser.add_argument('--m_iter', type = int, default = 1000)
parser.add_argument('--log_loss', type = str, default = 'False')
parser.add_argument('--log_sharp', type = str, default = 'False')
parser.add_argument('--log_pre_sharp', type = str, default = 'True')

config = parser.parse_args()


# Model parameters
config.model = f'fcn_{config.abc}'
config.use_bias = True if config.bias == 'True' else False
config.use_augment = True if config.augment == 'True' else False

save_dir = 'fcn_results'

warm_lst = [1, 64, 256, 1024]
palette = sns.color_palette('tab10', n_colors = len(warm_lst))
colors = {warm_lst[i]:palette[i] for i in range(len(warm_lst))}

metrics = ['train_loss', 'train_accuracy', 'test_loss', 'test_accuracy']
titles = {'train_loss': 'Train loss', 'train_accuracy': 'Train accuracy', 'test_loss': 'Test loss', 'test_accuracy': 'Test accuracy'}

dfs_train = list()
dfs_eval = list()


for warmup_steps in warm_lst:
    config.warmup_steps = warmup_steps
    divergence = False
    
    train_path = f'{save_dir}/train_{config.dataset}_{config.model}_scale{config.scale}_varw{config.varw}_n{config.width}_d{config.depth}_bias{config.use_bias}_{config.act_name}_I{config.init_seed}_{config.loss_name}_augment{config.augment}_{config.opt_name}_lr{config.lr_trgt}_a{config.warmup_exponent}_b{config.decay_exponent}_Twarm{config.warmup_steps}_T{config.num_steps}_B{config.batch_size}_b1{config.b1}_b2{config.b2}_{config.sharpness_method}.tab'    
    eval_path = f'{save_dir}/eval_{config.dataset}_{config.model}_scale{config.scale}_varw{config.varw}_n{config.width}_d{config.depth}_bias{config.use_bias}_{config.act_name}_I{config.init_seed}_{config.loss_name}_augment{config.augment}_{config.opt_name}_lr{config.lr_trgt}_a{config.warmup_exponent}_b{config.decay_exponent}_Twarm{config.warmup_steps}_T{config.num_steps}_B{config.batch_size}_b1{config.b1}_b2{config.b2}_{config.sharpness_method}.tab'    
    try:
        # create a dataframe
        df = pd.read_csv(train_path, sep = '\t', index_col = 0)
        df['lr_trgt'] = df['lr'].max()
        df['warmup_steps'] = config.warmup_steps
        max_train_accuracy = df.iloc[-1]['accuracy_step']
        print(max_train_accuracy)
        dfs_train.append(df)

        df = pd.read_csv(eval_path, sep = '\t', index_col = 0)
        df['lr_trgt'] = df['lr'].max()
        df['warmup_steps'] = config.warmup_steps; df['2lr'] = 2 / df['lr']
        # save training data
        dfs_eval.append(df)
        
    except FileNotFoundError:
        print(f'Divergence', train_path)
        
thresh = (2 + 2 * config.b1) / (1 - config.b1)

dfs_train = pd.concat(dfs_train, axis = 0, ignore_index = True)
dfs_train['clr'] = thresh / dfs_train['lr']
print(dfs_train.columns)

dfs_eval = pd.concat(dfs_eval, axis = 0, ignore_index = True)
dfs_eval['clr'] = thresh / dfs_eval['lr']
print(dfs_eval.columns)

sharpness_max = dfs_train['sharpness_step'].max()
pre_sharpness_max = dfs_train['pre_sharpness_step'].max()

print('Initial sharpness: ', dfs_train.iloc[0]['sharpness_step'])


## TRAINING LOSS

fig, ax = plt.subplots(1, 1, figsize = (10, 6.5), sharex = True)
ax.grid()
fig.subplots_adjust(right = 0.85)

ax = sns.lineplot(x = 'step', y = 'loss_step', data = dfs_train, hue = 'warmup_steps', ax = ax, palette = colors)
ax.set_xlabel('step')
ax.set_ylabel(r'Training loss')
ax.legend(title = r'$T_{\mathrm{wrm}}$', fontsize = 22)

ax.set_xscale('log', base = 10)
num_ticks = np.int_(np.log10(warm_lst[-1])) + 1
x_ticks = [10**i for i in range(num_ticks)] 
x_tick_labels = ['$10^{}$'.format(i) for i in range(num_ticks)]  # LaTeX formatted labels
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_tick_labels)

if config.log_loss == 'True':
    ax.set_yscale('log')

path = f'plots/fcns_adam/loss_{config.dataset}_{config.model}_s{config.scale}_varw{config.varw}_n{config.width}_d{config.depth}_{config.act_name}_{config.loss_name}_augment{config.augment}_{config.opt_name}_lr{config.lr_trgt}_a{config.warmup_exponent}_b{config.decay_exponent}_Tw{config.warmup_steps}_T{config.num_steps}_B{config.batch_size}_b1{config.b1}_b2{config.b2}.pdf'
fig.savefig(path, dpi = 300, bbox_inches = 'tight')

plt.show()


## SHARPNESS

fig, ax = plt.subplots(1, 1, figsize = (10, 6.5), sharex = True)
ax.grid()
fig.subplots_adjust(right = 0.85)

ax = sns.lineplot(x = 'step', y = 'pre_sharpness_step', data = dfs_train, hue = 'warmup_steps', ax = ax, palette = colors)
ax = sns.lineplot(x = 'step', y = 'clr', data = dfs_train, hue = 'warmup_steps', ax = ax, palette = colors, legend = False, linestyle = 'dashed')

ax.set_xlabel('step')
ax.set_ylabel(r'$\lambda_t^{P^{-1}H}$')
ax.legend(title = r'$T_{\mathrm{wrm}}$', fontsize = 22)

num_ticks = np.int_(np.log10(config.num_steps)) + 1
x_ticks = [10**i for i in range(num_ticks)] 
x_tick_labels = ['$10^{}$'.format(i) for i in range(num_ticks)]  # LaTeX formatted labels
ax.set_xscale('log', base = 10)
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_tick_labels)

if config.log_pre_sharp == 'True':
    ax.set_yscale('log')

path = f'plots/fcns_adam/pre_sharp_{config.dataset}_{config.model}_s{config.scale}_varw{config.varw}_n{config.width}_d{config.depth}_{config.act_name}_{config.loss_name}_augment{config.augment}_{config.opt_name}_lr{config.lr_trgt}_a{config.warmup_exponent}_b{config.decay_exponent}_Tw{config.warmup_steps}_T{config.num_steps}_B{config.batch_size}_b1{config.b1}_b2{config.b2}.pdf'
fig.savefig(path, dpi = 300, bbox_inches = 'tight')
plt.show()


fig, ax = plt.subplots(1, 1, figsize = (10, 6.5), sharex = True)
ax.grid()
fig.subplots_adjust(right = 0.85)

ax = sns.lineplot(x = 'step', y = 'sharpness_step', data = dfs_train, hue = 'warmup_steps', ax = ax, palette = colors)
ax.set_xlabel('step')
ax.set_ylabel(r'$\lambda_t^H$')
ax.legend(title = r'$T_{\mathrm{wrm}}$', fontsize = 22)

num_ticks = np.int_(np.log10(config.num_steps)) + 1
x_ticks = [10**i for i in range(num_ticks)] 
x_tick_labels = ['$10^{}$'.format(i) for i in range(num_ticks)]  # LaTeX formatted labels
ax.set_xscale('log', base = 10)
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_tick_labels)

#ax.set_ylim(0, 1.5*pre_sharpness_max)
if config.log_sharp == 'True':
    ax.set_yscale('log')

path = f'plots/fcns_adam/sharp_{config.dataset}_{config.model}_s{config.scale}_varw{config.varw}_n{config.width}_d{config.depth}_{config.act_name}_{config.loss_name}_augment{config.augment}_{config.opt_name}_lr{config.lr_trgt}_a{config.warmup_exponent}_b{config.decay_exponent}_Tw{config.warmup_steps}_T{config.num_steps}_B{config.batch_size}_b1{config.b1}_b2{config.b2}.pdf'
fig.savefig(path, dpi = 300, bbox_inches = 'tight')
plt.show()

fig, ax = plt.subplots(1, 1, figsize = (10, 6.5), sharex = True)
ax.grid()
fig.subplots_adjust(right = 0.85)

ax = sns.lineplot(x = 'step', y = 'grads_norm_step', data = dfs_train, hue = 'warmup_steps', ax = ax, palette = colors)
ax.set_xlabel('step')
ax.set_ylabel(r'$\|g_t\|$')
ax.legend(title = r'$T_{\mathrm{wrm}}$', fontsize = 22)

num_ticks = np.int_(np.log10(config.num_steps)) + 1
x_ticks = [10**i for i in range(num_ticks)] 
x_tick_labels = ['$10^{}$'.format(i) for i in range(num_ticks)]  # LaTeX formatted labels
ax.set_xscale('log', base = 10)
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_tick_labels)

#ax.set_ylim(0, 1.5*pre_sharpness_max)
if config.log_sharp == 'True':
    ax.set_yscale('log')

path = f'plots/fcns_adam/grads_{config.dataset}_{config.model}_s{config.scale}_varw{config.varw}_n{config.width}_d{config.depth}_{config.act_name}_{config.loss_name}_augment{config.augment}_{config.opt_name}_lr{config.lr_trgt}_a{config.warmup_exponent}_b{config.decay_exponent}_Tw{config.warmup_steps}_T{config.num_steps}_B{config.batch_size}_b1{config.b1}_b2{config.b2}.pdf'
fig.savefig(path, dpi = 300, bbox_inches = 'tight')
plt.show()

fig, ax = plt.subplots(1, 1, figsize = (10, 6.5), sharex = True)
ax.grid()
fig.subplots_adjust(right = 0.85)

ax = sns.lineplot(x = 'step', y = 'mu_norm_step', data = dfs_train, hue = 'warmup_steps', ax = ax, palette = colors)
ax.set_xlabel('step')
ax.set_ylabel(r'$\|m_t\|$')
ax.legend(title = r'$T_{\mathrm{wrm}}$', fontsize = 22)

num_ticks = np.int_(np.log10(config.num_steps)) + 1
x_ticks = [10**i for i in range(num_ticks)] 
x_tick_labels = ['$10^{}$'.format(i) for i in range(num_ticks)]  # LaTeX formatted labels
ax.set_xscale('log', base = 10)
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_tick_labels)

#ax.set_ylim(0, 1.5*pre_sharpness_max)
if config.log_loss == 'True':
    ax.set_yscale('log')

path = f'plots/fcns_adam/mu_{config.dataset}_{config.model}_s{config.scale}_varw{config.varw}_n{config.width}_d{config.depth}_{config.act_name}_{config.loss_name}_augment{config.augment}_{config.opt_name}_lr{config.lr_trgt}_a{config.warmup_exponent}_b{config.decay_exponent}_Tw{config.warmup_steps}_T{config.num_steps}_B{config.batch_size}_b1{config.b1}_b2{config.b2}.pdf'
fig.savefig(path, dpi = 300, bbox_inches = 'tight')
plt.show()

fig, ax = plt.subplots(1, 1, figsize = (10, 6.5), sharex = True)
ax.grid()
fig.subplots_adjust(right = 0.85)

ax = sns.lineplot(x = 'step', y = 'nu_norm_step', data = dfs_train, hue = 'warmup_steps', ax = ax, palette = colors)
ax.set_xlabel('step')
ax.set_ylabel(r'$\|v_t\|$')
ax.legend(title = r'$T_{\mathrm{wrm}}$', fontsize = 22)

num_ticks = np.int_(np.log10(config.num_steps)) + 1
x_ticks = [10**i for i in range(num_ticks)] 
x_tick_labels = ['$10^{}$'.format(i) for i in range(num_ticks)]  # LaTeX formatted labels
ax.set_xscale('log', base = 10)
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_tick_labels)

#ax.set_ylim(0, 1.5*pre_sharpness_max)
if config.log_sharp == 'True':
    ax.set_yscale('log')

path = f'plots/fcns_adam/nu_{config.dataset}_{config.model}_s{config.scale}_varw{config.varw}_n{config.width}_d{config.depth}_{config.act_name}_{config.loss_name}_augment{config.augment}_{config.opt_name}_lr{config.lr_trgt}_a{config.warmup_exponent}_b{config.decay_exponent}_Tw{config.warmup_steps}_T{config.num_steps}_B{config.batch_size}_b1{config.b1}_b2{config.b2}.pdf'
fig.savefig(path, dpi = 300, bbox_inches = 'tight')
plt.show()



## NUMBER OF ITERATIONS

fig, ax = plt.subplots(1, 1, figsize = (10, 6.5), sharex = True)
ax.grid()
fig.subplots_adjust(right = 0.85)

ax = sns.lineplot(x = 'step', y = 'n_iter', data = dfs_train, hue = 'warmup_steps', ax = ax, palette = colors)
ax.set_xlabel('step')
ax.set_ylabel(r'$n_{\mathrm{iter}}$')
ax.legend(title = r'$T_{\mathrm{wrm}}$', fontsize = 22)

num_ticks = np.int_(np.log10(config.num_steps)) + 1
x_ticks = [10**i for i in range(num_ticks)] 
x_tick_labels = ['$10^{}$'.format(i) for i in range(num_ticks)]  # LaTeX formatted labels
ax.set_xscale('log', base = 10)
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_tick_labels)
path = f'plots/fcns_adam/niter_{config.dataset}_{config.model}_s{config.scale}_varw{config.varw}_n{config.width}_d{config.depth}_{config.act_name}_{config.loss_name}_augment{config.augment}_{config.opt_name}_lr{config.lr_trgt}_a{config.warmup_exponent}_b{config.decay_exponent}_Tw{config.warmup_steps}_T{config.num_steps}_B{config.batch_size}_b1{config.b1}_b2{config.b2}.pdf'
# fig.savefig(path, dpi = 300, bbox_inches = 'tight')
plt.show()


## NUMBER OF ITERATIONS

fig, ax = plt.subplots(1, 1, figsize = (10, 6.5), sharex = True)
ax.grid()
fig.subplots_adjust(right = 0.85)

ax = sns.lineplot(x = 'step', y = 'pre_n_iter', data = dfs_train, hue = 'warmup_steps', ax = ax, palette = colors)
ax.set_xlabel('step')
ax.set_ylabel(r'pre $n_{\mathrm{iter}}$')
ax.legend(title = r'$T_{\mathrm{wrm}}$', fontsize = 22)

num_ticks = np.int_(np.log10(config.num_steps)) + 1
x_ticks = [10**i for i in range(num_ticks)] 
x_tick_labels = ['$10^{}$'.format(i) for i in range(num_ticks)]  # LaTeX formatted labels
ax.set_xscale('log', base = 10)
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_tick_labels)

ax.set_yscale('log', base = 10)
path = f'plots/fcns_adam/pre_niter_{config.dataset}_{config.model}_s{config.scale}_varw{config.varw}_n{config.width}_d{config.depth}_{config.act_name}_{config.loss_name}_augment{config.augment}_{config.opt_name}_lr{config.lr_trgt}_a{config.warmup_exponent}_b{config.decay_exponent}_Tw{config.warmup_steps}_T{config.num_steps}_B{config.batch_size}_b1{config.b1}_b2{config.b2}.pdf'
# fig.savefig(path, dpi = 300, bbox_inches = 'tight')
plt.show()

