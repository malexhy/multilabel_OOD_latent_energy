import torch
from OOD_model import get_model_fe_only
import json
import argparse
from dataset.dataset import get_dataLoaders, get_standard_test_transform
from tqdm import tqdm
import os
from torch_pca import PCA
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import silhouette_samples
from sklearn.metrics.pairwise import pairwise_distances

    
def get_fv_list(model, data_loader, device='cuda'):
    fv_list = []
    
    model.eval()
    model.to(device)
    
    with torch.no_grad():
        for img, *_ in tqdm(data_loader, disable=False, bar_format="{l_bar}{bar:30}{r_bar}"):
            img = img.to(device)
            fv = model(img)
            fv_list.append(fv.detach().cpu())
    
    fv_list = torch.cat(fv_list, dim=0)
    return fv_list

def reduce_fv_low_dim_pca(fv_list, dim):
    reducer = PCA(n_components=dim, svd_solver='full')
    reduced_latent = reducer.fit_transform(fv_list)
    return reduced_latent


def plot_data(data, save_path, dim=2):
    base_size = 10
    fig = plt.figure(figsize=(base_size, base_size))
    grid = gridspec.GridSpec(ncols=1, nrows=1, figure=fig,)
    if dim == 2:
        ax = fig.add_subplot(grid[0,0])
        ax.set_title('latent', fontsize = base_size*5)
        ax.scatter(data[:,0], data[:,1], marker='o', facecolors='none', edgecolors='blue')
    elif dim == 3:
        ax = fig.add_subplot(grid[0,0],projection='3d')
        ax.scatter(data[:,0], data[:,1], data[:,2], marker='o', facecolors='none', edgecolors='blue')
        
    fig.tight_layout()
    fig.savefig(save_path)
    plt.clf()
    
def plot_data_group(data, group, save_path, dim=2, plot_outliner=False):
    base_size = 10
    fig = plt.figure(figsize=(base_size, base_size))
    grid = gridspec.GridSpec(ncols=1, nrows=1, figure=fig,)
    # prop_cycle = plt.rcParams['axes.prop_cycle']
    # colors = prop_cycle.by_key()['color']
    n_group = group.max() + 1
    colors = plt.cm.hsv(torch.linspace(0, 1, n_group))
    if dim == 2:
        ax = fig.add_subplot(grid[0,0])
        # ax.set_title('latent', fontsize = base_size*5)
        if plot_outliner:
            target = (group == -1)    
            td = data[target,:]
            ax.scatter(td[:,0], td[:,1], marker='o', facecolors='none', edgecolors='gray')
            ax.set_axis_off()
        for i in range(n_group):
            target = (group == i)
            td = data[target,:]
            # ax.scatter(td[:,0], td[:,1], marker='o', facecolors='none', edgecolors=colors[i])
            ax.scatter(td[:,0], td[:,1], marker='o', facecolors=colors[i])
        
    elif dim == 3:
        ax = fig.add_subplot(grid[0,0],projection='3d')
        target = (group == -1)    
        td = data[target,:]
        ax.scatter(td[:,0], td[:,1], td[:,2], marker='o', facecolors='none', edgecolors='gray')
        for i in range(n_group):
            target = (group == i)
            td = data[target,:]
            ax.scatter(td[:,0], td[:,1], td[:,2], marker='o', facecolors='none', edgecolors=colors[i])
        
    fig.tight_layout()
    fig.savefig(save_path)
    plt.clf()
    plt.close()
        
        
def kmean_find_best_k(data, min_k=10, max_k=300, step=10, method=['ss'], early_stop_max=5, device='cpu', eval_data=None, n_init='auto', dist_matrix=None):
    best_kmeans = None
    best_k = None
    # best_score = -100.
    best_score = {
        'ss': -100.,
        'chs': -100., 
        'dbs': -100.,
    }
    best_score = {m:-100. for m in method}
    early_stop_count = {k:0 for k in method}
    _data = data if eval_data is None else eval_data
    if dist_matrix is None:
        dist_matrix = _data
        
    for n_clusters in range(min_k, max_k+step, step):
        stop = True
        for m in method:
            early_stop_count[m] += 1
            

        if device == 'cpu':
            cluster = KMeans(n_clusters=n_clusters, random_state=42, n_init=n_init, init='k-means++')
            kmeans = cluster.fit(data)
        elif device == 'cuda':
            print('Not support cuda Kmeans.')
            1/0


        print_log = f'k={n_clusters:<4}: '
        
        if 'ss' in method:
            ss = silhouette_score(dist_matrix, kmeans.labels_, metric='precomputed', n_jobs=-1)
            if best_score['ss'] < ss:
                best_score['ss'] = ss
                early_stop_count['ss'] = 0
            print_log += f'ss={ss:<.5f}, '
            
        if 'chs' in method:
            chs = calinski_harabasz_score(_data, kmeans.labels_)
            if best_score['chs'] < chs:
                best_score['chs'] = chs
                early_stop_count['chs'] = 0
            print_log += f'chs={chs:<.5f}, '
            
        if 'dbs' in method:
            dbs = -davies_bouldin_score(_data, kmeans.labels_)
            if best_score['dbs'] < dbs:
                best_score['dbs'] = dbs
                early_stop_count['dbs'] = 0
            print_log += f'db={dbs:<.5f}, '
            
        print(print_log)
        
        
        for m in method:
            # print(early_stop_count[m])
            if early_stop_count[m] <= early_stop_max:
                stop = False
                
            if early_stop_count[m] == 0:
                best_kmeans = kmeans
                best_k = n_clusters
                break
        
        
        if stop:
            print(best_kmeans)
            break
    
    
    return best_k, best_kmeans

    
def cluster(load_path, ckpt, save_group=True, use_split='val', cluster_method='kmeans', save_figure=False, sub_version=None):

    torch_save_path = load_path + '/' + ckpt

    
    if use_split == 'val' or use_split == 'train':
        fv_save_path = os.path.join(torch_save_path, f'fv_list_{use_split}.pt')
    else:
        1/0

    if not os.path.exists(fv_save_path):
    # if True:
        print(f'Computing fv list.')
        with open(f'{load_path}/params.txt', 'r') as f:
            log_dict = json.load(f)
        log_args = argparse.Namespace(**log_dict)
        ck_path = os.path.join(load_path, ckpt, 'fe.pt')
        if os.path.exists(ck_path):
            log_args.load_path = load_path + '/' + ckpt + '/fe.pt'
        else:
            print('Fe path not found, using imagenet-1k pretrain weight.')
            log_args.load_path = None
            
        model = get_model_fe_only(log_args, 'cpu')
        train_dl, val_dl = get_dataLoaders(log_args)
        if use_split == 'val':
            fv_list = get_fv_list(model, val_dl, 'cuda')
        elif use_split == 'train':
            test_transform = get_standard_test_transform(log_args)
            if 'nus' in log_args.training_set:
                train_dl.dataset.img_transform = test_transform
            elif 'voc' in log_args.training_set:
                train_dl.dataset.transform = test_transform
            elif 'coco' in log_args.training_set:
                train_dl.dataset.img_transform = test_transform
            else:
                1/0
            fv_list = get_fv_list(model, train_dl, 'cuda')
        print(f'Saving fv list to {fv_save_path}.')
        # exit()
        torch.save(fv_list, fv_save_path)
    else:
        print(f'Found exists fv list from {fv_save_path}. \nLoading......')
        with open(f'{load_path}/params.txt', 'r') as f:
            log_dict = json.load(f)
        log_args = argparse.Namespace(**log_dict)
        fv_list = torch.load(fv_save_path)

    fv_list_norm = fv_list/torch.norm(fv_list, dim=1, keepdim=True)

    if 'coco' in log_args.training_set:
        # print('Using coco')
        dim = 100 # coco
    elif 'voc' in log_args.training_set:
        # print('Using voc')
        dim = 100 # voc
    elif 'nus' in log_args.training_set:
        dim = 100
        
    fv_list_low_dim = reduce_fv_low_dim_pca(fv_list_norm, dim)    

    norm_data = fv_list_low_dim
    save_suffix = ''
    if 'kmeans' in cluster_method:
        n_clusters = cluster_method.replace('kmeans_', '')
        if 'voc' in log_args.training_set:
            step_size = 1
            early_stop_max = 10
            final_loop = 10

            bk_n_init = 5
            fi_n_init = 5
            bk_score = ['chs', 'ss']
            eval_data = None
            
        elif 'coco' in log_args.training_set:
            step_size = 1
            early_stop_max = 5
            final_loop = 10
            # final_loop = 1
            bk_n_init = 5
            fi_n_init = 5
            bk_score = ['chs', 'ss']
            eval_data = None
            
        elif 'nus' in log_args.training_set:
            if '_' in log_args.training_set:
                step_size = 1
            else:
                step_size = 5
            early_stop_max = 5
            final_loop = 5
            bk_n_init = 5
            fi_n_init = 'auto'
            bk_score = ['chs', 'ss']
            eval_data = None
        
        if eval_data is None:
            eval_data = norm_data
        
        fi_score = 'chs' 
        fi_init = 'random'
        
        if len(eval_data) < 2000:
            filter_outliner = False
        else:
            filter_outliner = True
            
        dist_matrix = pairwise_distances(eval_data, metric='euclidean', n_jobs=-1)
        
        if n_clusters == 'auto':

            min_k = round(log_args.n_classes, -1)

            k, _ = kmean_find_best_k(norm_data.clone(), method=bk_score, min_k=min_k, step=step_size, early_stop_max=early_stop_max, device='cpu', eval_data=eval_data, n_init=bk_n_init, dist_matrix=dist_matrix)

            save_suffix = 'kmeans_auto'
            
            
        else:
            k = int(n_clusters.replace('_', ''))
            save_suffix = f'kmeans_{k}'
            

        use_best = True
        highest_score = 0
        worse_score = 10000
        kmeans = None
        for _ in range(final_loop):

            clusterer = KMeans(n_clusters=k, random_state=None, n_init=fi_n_init, init=fi_init, max_iter=1000)
            _kmeans = clusterer.fit(norm_data)
            if fi_score == 'ss':
                score = silhouette_score(dist_matrix, _kmeans.labels_, metric='precomputed', n_job=-1)
            elif fi_score == 'chs':
                score = calinski_harabasz_score(eval_data, _kmeans.labels_)
            else:
                print(f'Not support {fi_score}')
                1/0
            print(f'{fi_score}: {score:.2f}')
            if score > highest_score:
                highest_score = score
                kmeans = _kmeans
                
            if score < worse_score:
                worse_score = score
                w_kmeans = _kmeans
        
        if use_best:
            cluster_labels = kmeans.labels_
        else:
            cluster_labels = w_kmeans.labels_
        
        if filter_outliner:
            sil_thr = -0.2
            # sil_thr = 0.
            sil_scores = silhouette_samples(dist_matrix, cluster_labels, metric='precomputed', n_job=-1)
            while True:
                outliers_sil = sil_scores < sil_thr
                temp_label = cluster_labels.copy()
                temp_label[outliers_sil] = -1
                is_complete = len(set(range(cluster_labels.max()+1)) - set(temp_label)) == 0
                if is_complete:
                    cluster_labels = temp_label
                    print(f'N outliner: {(cluster_labels==-1).sum()}')  
                    break
                else:
                    miss_label = set(range(cluster_labels.max()+1)) - set(temp_label)
                    sil_thr -= 0.05

        if use_best:
            print(f'Best cluster: K = {cluster_labels.max()+1}, {fi_score} = {highest_score:.2f}')
        else:
            print(f'Best cluster: K = {cluster_labels.max()+1}, {fi_score} = {worse_score:.2f}')

    
    if save_group:
        cluster_labels_tensor = torch.tensor(cluster_labels)
        if use_split == 'val' or  use_split == 'train':
            group_path = os.path.join(torch_save_path, f'latent_group_{use_split}_{save_suffix}')
            if sub_version is not None:
                group_path += f'_{sub_version}'
            group_path += '.pt'
                
        else:
            1/0
        torch.save(cluster_labels_tensor, group_path)

    if save_figure:
        fv_list_3d = reduce_fv_low_dim_pca(fv_list_norm, 3)
        plot_data_group(
            data=fv_list_3d, 
            group=cluster_labels,
            save_path=os.path.join(load_path, f'latent_grouped_pca_3D_{use_split}.png'), 
            dim=3,
            plot_outliner=True,)

        fv_list_2d = reduce_fv_low_dim_pca(fv_list_norm, 2)
        plot_data_group(
            data=fv_list_2d, 
            group=cluster_labels,
            save_path=os.path.join(load_path, f'latent_grouped_pca_2D_{use_split}.png'), 
            dim=2,
            plot_outliner=True,)




