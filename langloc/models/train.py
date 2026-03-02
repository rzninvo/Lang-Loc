"""Training pipeline with contrastive loss and k-fold cross-validation."""

import time
from pathlib import Path
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import wandb
import random

import hydra
from omegaconf import DictConfig, OmegaConf

from langloc.data_processing.scene_graph import SceneGraph
from langloc.analysis.helper import get_matching_subgraph
from langloc.models.model_graph2graph import BigGNN
from langloc.models.train_utils import cross_entropy, k_fold_by_scene


def format_to_latex(acc):
    """Formats an accuracy dictionary as LaTeX-style percentage strings.

    Args:
        acc: Dictionary mapping top-k values to ``(mean, std)`` tuples.

    Returns:
        Multi-line string with each key formatted as ``$mean \\pm std$``.
    """
    acc_string = ''
    for k, v in acc.items():
        acc_string += f'{k}: ${v[0] * 100:.2f} \pm {v[1] * 100:.2f}$\n'
    return acc_string


def train(model, optimizer, database_3dssg, dataset, batch_size, fold, cfg, device='cuda'):
    """Runs one epoch of contrastive training over batched graph pairs.

    For each batch, computes pairwise cosine similarity and matching probability
    losses between query graphs and their corresponding database scenes.

    Args:
        model: BigGNN model to train.
        optimizer: Torch optimizer.
        database_3dssg: Dictionary mapping scene IDs to 3DSSG SceneGraph objects.
        dataset: List of query SceneGraph objects.
        batch_size: Number of graphs per batch.
        fold: Current fold index (for wandb logging).
        cfg: Hydra DictConfig.

    Returns:
        The trained model.
    """
    assert type(dataset) == list, "dataset must be a list"
    indices = [i for i in range(len(dataset))]
    random.shuffle(indices)
    if (cfg.train.contrastive_loss):
        batched_indices = [indices[i:i+batch_size] for i in range(0, len(indices) - batch_size, batch_size)]
        assert len(batched_indices[0]) == batch_size, "First batch must be full-sized"
        skipped = 0
        total = 0
        for batch in batched_indices:
            loss1 = torch.zeros((len(batch), len(batch))).to(device)
            loss3 = torch.zeros((len(batch), len(batch))).to(device)
            for i in range(len(batch)):
                for j in range(i, len(batch)):
                    total += 1
                    query = dataset[batch[i]]
                    db = database_3dssg[dataset[batch[j]].scene_id]
                    if (cfg.train.subgraph_ablation):
                        query_subgraph, db_subgraph = query, db
                    else:
                        query_subgraph, db_subgraph = get_matching_subgraph(query, db, cfg.graph.dbscan_eps, cfg.graph.dbscan_min_samples)
                        if db_subgraph is None or len(db_subgraph.nodes) <= 1: db_subgraph = db
                        if query_subgraph is None or len(query_subgraph.nodes) <= 1: query_subgraph = query

                    x_node_ft, x_edge_idx, x_edge_ft = query_subgraph.to_pyg()
                    p_node_ft, p_edge_idx, p_edge_ft = db_subgraph.to_pyg()
                    if len(x_edge_idx[0]) < 1 or len(p_edge_idx[0]) < 1:
                        skipped += 1
                        loss1[i][j] = 1
                        loss1[j][i] = loss1[i][j]
                        loss3[i][j] = 0.5
                        loss3[j][i] = loss3[i][j]
                        continue
                    x_p, p_p, m_p = model(torch.tensor(np.array(x_node_ft), dtype=torch.float32).to(device), torch.tensor(np.array(p_node_ft), dtype=torch.float32).to(device),
                                            torch.tensor(x_edge_idx, dtype=torch.int64).to(device), torch.tensor(p_edge_idx, dtype=torch.int64).to(device),
                                            torch.tensor(np.array(x_edge_ft), dtype=torch.float32).to(device), torch.tensor(np.array(p_edge_ft), dtype=torch.float32).to(device))
                    x_node_ft, x_edge_idx, x_edge_ft = None, None, None

                    loss1[i][j] = 1 - F.cosine_similarity(x_p, p_p, dim=0)
                    loss1[j][i] = loss1[i][j]
                    loss3[i][j] = m_p
                    loss3[j][i] = loss3[i][j]
            loss1_t = (torch.ones((len(batch), len(batch))).to(device) - torch.eye(len(batch)).to(device)) * 2
            loss3_t = torch.eye(len(batch)).to(device)

            avg_mp = torch.diag(loss3).mean()
            avg_mn = (torch.sum(loss3) - torch.diag(loss3).sum()) / (len(batch) * (len(batch) - 1))
            avg_cos_sim_p = torch.diag(loss1).mean()
            avg_cos_sim_n = (torch.sum(loss1) - torch.diag(loss1).sum()) / (len(batch) * (len(batch) - 1))

            loss1 = cross_entropy(loss1, loss1_t, reduction='mean', dim=1)
            loss3 = cross_entropy(loss3, loss3_t, reduction='mean', dim=1)
            if (cfg.train.loss_ablation_m): loss = loss1
            elif (cfg.train.loss_ablation_c): loss = loss3
            else:
                w_c = cfg.train.loss_weight_cosine
                w_m = cfg.train.loss_weight_matching
                loss = (w_c * loss1 + w_m * loss3) / (w_c + w_m)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            wandb.log({f'loss1_{fold}': loss1.item(),
                        f'loss3_{fold}': loss3.item(),
                        f'loss_{fold}': loss.item(),
                        f'avg_matching_pos_{fold}': avg_mp.item(),
                        f'avg_matching_neg_{fold}': avg_mn.item(),
                        f'avg_cos_sim_pos_{fold}': avg_cos_sim_p.item(),
                        f'avg_cos_sim_neg_{fold}': avg_cos_sim_n.item()})
        print(f'Skipped {skipped} graphs out of {total} because one of the subgraphs had too few edges')
    return model


def eval_loss(model, database_3dssg, dataset, fold, cfg, device='cuda'):
    """Evaluates the model loss on a validation dataset.

    Computes contrastive loss (cosine similarity + matching probability)
    across batches without gradient updates.

    Args:
        model: BigGNN model to evaluate.
        database_3dssg: Dictionary mapping scene IDs to 3DSSG SceneGraph objects.
        dataset: List of validation SceneGraph objects.
        fold: Current fold index (for wandb logging).
        cfg: Hydra DictConfig.

    Returns:
        Mean loss across all batches.
    """
    model.eval()
    loss1_across_batches = []
    loss3_across_batches = []
    loss_across_batches = []
    avg_mp_across_batches = []
    avg_mn_across_batches = []
    avg_cos_sim_p_across_batches = []
    avg_cos_sim_n_across_batches = []
    with torch.no_grad():
        assert type(dataset) == list, "dataset must be a list"
        indices = [i for i in range(len(dataset))]
        random.shuffle(indices)
        if (cfg.train.contrastive_loss):
            batched_indices = [indices[i:i+cfg.train.batch_size] for i in range(0, len(indices) - cfg.train.batch_size, cfg.train.batch_size)]
            assert len(batched_indices[0]) == cfg.train.batch_size, "First batch must be full-sized"
            print(f'number of batches in evaluation: {len(batched_indices)}')
            skipped = 0
            total = 0
            for batch in batched_indices:
                loss1 = torch.zeros((len(batch), len(batch))).to(device)
                loss3 = torch.zeros((len(batch), len(batch))).to(device)
                for i in range(len(batch)):
                    for j in range(i, len(batch)):
                        total += 1
                        query = dataset[batch[i]]
                        db = database_3dssg[dataset[batch[j]].scene_id]
                        if (cfg.train.subgraph_ablation):
                            query_subgraph, db_subgraph = query, db
                        else:
                            query_subgraph, db_subgraph = get_matching_subgraph(query, db, cfg.graph.dbscan_eps, cfg.graph.dbscan_min_samples)
                            if db_subgraph is None or len(db_subgraph.nodes) <= 1: db_subgraph = db
                            if query_subgraph is None or len(query_subgraph.nodes) <= 1: query_subgraph = query

                        x_node_ft, x_edge_idx, x_edge_ft = query_subgraph.to_pyg()
                        p_node_ft, p_edge_idx, p_edge_ft = db_subgraph.to_pyg()
                        if len(x_edge_idx[0]) < 1 or len(p_edge_idx[0]) < 1:
                            skipped += 1
                            loss1[i][j] = 1
                            loss1[j][i] = loss1[i][j]
                            loss3[i][j] = 0.5
                            loss3[j][i] = loss3[i][j]
                            continue
                        x_p, p_p, m_p = model(torch.tensor(np.array(x_node_ft), dtype=torch.float32).to(device), torch.tensor(np.array(p_node_ft), dtype=torch.float32).to(device),
                                                torch.tensor(x_edge_idx, dtype=torch.int64).to(device), torch.tensor(p_edge_idx, dtype=torch.int64).to(device),
                                                torch.tensor(np.array(x_edge_ft), dtype=torch.float32).to(device), torch.tensor(np.array(p_edge_ft), dtype=torch.float32).to(device))
                        x_node_ft, x_edge_idx, x_edge_ft = None, None, None
                        loss1[i][j] = 1 - F.cosine_similarity(x_p, p_p, dim=0)
                        loss1[j][i] = loss1[i][j]
                        loss3[i][j] = m_p
                        loss3[j][i] = loss3[i][j]
                loss1_t = (torch.ones((len(batch), len(batch))).to(device) - torch.eye(len(batch)).to(device)) * 2
                loss3_t = torch.eye(len(batch)).to(device)

                avg_mp = torch.diag(loss3).mean()
                avg_mn = (torch.sum(loss3) - torch.diag(loss3).sum()) / (len(batch) * (len(batch) - 1))
                avg_cos_sim_p = torch.diag(loss1).mean()
                avg_cos_sim_n = (torch.sum(loss1) - torch.diag(loss1).sum()) / (len(batch) * (len(batch) - 1))

                loss1 = cross_entropy(loss1, loss1_t, reduction='mean', dim=1)
                loss3 = cross_entropy(loss3, loss3_t, reduction='mean', dim=1)
                if (cfg.train.loss_ablation_m or cfg.eval.eval_only_c): loss = loss1
                elif (cfg.train.loss_ablation_c): loss = loss3
                else:
                    w_c = cfg.train.loss_weight_cosine
                    w_m = cfg.train.loss_weight_matching
                    loss = (w_c * loss1 + w_m * loss3) / (w_c + w_m)

                loss1_across_batches.append(loss1.item())
                loss3_across_batches.append(loss3.item())
                loss_across_batches.append(loss.item())
                avg_mp_across_batches.append(avg_mp.item())
                avg_mn_across_batches.append(avg_mn.item())
                avg_cos_sim_p_across_batches.append(avg_cos_sim_p.item())
                avg_cos_sim_n_across_batches.append(avg_cos_sim_n.item())

            wandb.log({f'eval_across_batch_loss1_{fold}': np.mean(loss1_across_batches),
                        f'eval_across_batch_loss3_{fold}': np.mean(loss3_across_batches),
                        f'eval_across_batch_loss_{fold}': np.mean(loss_across_batches),
                        f'eval_across_batch_avg_matching_pos_{fold}': np.mean(avg_mp_across_batches),
                        f'eval_across_batch_avg_matching_neg_{fold}': np.mean(avg_mn_across_batches),
                        f'eval_across_batch_avg_cos_sim_pos_{fold}': np.mean(avg_cos_sim_p_across_batches),
                        f'eval_across_batch_avg_cos_sim_neg_{fold}': np.mean(avg_cos_sim_n_across_batches)})
            print(f'During evaluation fold {fold} skipped {skipped} graphs out of {total} because one of the subgraphs had too few edges')
            print(f'Loss across batches was {np.mean(loss_across_batches)}')
    model.train()
    return torch.tensor(loss_across_batches).mean().item()


def eval_acc(model, database_3dssg, dataset, fold, cfg, mode='scanscribe', eval_iter_count=None, out_of=None, valid_top_k=[1, 2, 3, 5], timer=None, device='cuda'):
    """Evaluates top-k retrieval accuracy by sampling scene subsets.

    For each evaluation iteration, samples ``out_of`` scenes, scores the query
    against all of them, ranks by score, and checks if the ground-truth scene
    appears in the top-k.

    Args:
        model: BigGNN model to evaluate.
        database_3dssg: Dictionary mapping scene IDs to 3DSSG SceneGraph objects.
        dataset: List of query SceneGraph objects.
        fold: Current fold index (for wandb logging), or None.
        cfg: Hydra DictConfig.
        mode: Evaluation mode string for wandb logging (e.g. ``'scanscribe'``).
        eval_iter_count: Number of sample sets per eval iteration (overrides config).
        out_of: Number of candidate scenes per sample set (overrides config).
        valid_top_k: List of k values for top-k accuracy.
        timer: Optional Timer instance for benchmarking.

    Returns:
        Dictionary mapping each k to ``(mean_accuracy, std_accuracy)``.
    """
    if eval_iter_count is None:
        eval_iter_count = cfg.eval.eval_iter_count
    if out_of is None:
        out_of = cfg.eval.out_of
    model.eval()

    # Group dataset indices by scene_id
    buckets = {}
    for idx, g in enumerate(dataset):
        if g.scene_id not in buckets: buckets[g.scene_id] = []
        buckets[g.scene_id].append(idx)

    if cfg.eval.eval_entire_dataset:
        out_of = len(buckets)
        valid_top_k = [1, 5, 10, 20, 30, 40]
        if mode == 'human' or mode == 'human_test':
            valid_top_k.extend([50, 75])

    all_valid = {}
    for _ in range(cfg.eval.eval_iters):
        valid = {k: [] for k in valid_top_k}

        sampled_test_indices = [[random.sample(buckets[g], 1)[0] for g in random.sample(list(buckets.keys()), out_of)] for _ in range(eval_iter_count)]
        assert len(sampled_test_indices[0]) == out_of, "Sample set size must equal out_of"
        assert len(sampled_test_indices) == eval_iter_count, "Must have eval_iter_count sample sets"
        assert len(dataset) > 10, "Dataset must have more than 10 graphs"

        scene_ids_tset = []
        for t_set in sampled_test_indices:
            true_match = []
            match_prob = []
            cos_sims = []
            scene_ids_tset = []
            for i in t_set:
                query = dataset[t_set[0]]
                db = database_3dssg[dataset[i].scene_id]
                scene_ids_tset.append(db.scene_id)
                assert (query.scene_id == db.scene_id if i == t_set[0] else query.scene_id != db.scene_id), \
                    "First element must be ground-truth match"
                if (cfg.train.subgraph_ablation):
                    query_subgraph, db_subgraph = query, db
                else:
                    query_subgraph, db_subgraph = get_matching_subgraph(query, db, cfg.graph.dbscan_eps, cfg.graph.dbscan_min_samples)
                    if db_subgraph is None or len(db_subgraph.nodes) <= 1 or len(db_subgraph.edge_idx[0]) < 1: db_subgraph = db
                    if query_subgraph is None or len(query_subgraph.nodes) <= 1 or len(query_subgraph.edge_idx[0]) < 1: query_subgraph = query
                x_node_ft, x_edge_idx, x_edge_ft = query_subgraph.to_pyg()
                p_node_ft, p_edge_idx, p_edge_ft = db_subgraph.to_pyg()

                t1 = time.time()
                x_p, p_p, m_p = model(torch.tensor(np.array(x_node_ft), dtype=torch.float32).to(device), torch.tensor(np.array(p_node_ft), dtype=torch.float32).to(device),
                                        torch.tensor(x_edge_idx, dtype=torch.int64).to(device), torch.tensor(p_edge_idx, dtype=torch.int64).to(device),
                                        torch.tensor(np.array(x_edge_ft), dtype=torch.float32).to(device), torch.tensor(np.array(p_edge_ft), dtype=torch.float32).to(device))
                if timer is not None:
                    timer.text2graph_text_embedding_matching_score_time.append(time.time() - t1)
                    timer.text2graph_text_embedding_matching_score_iter.append(1)

                cos_sims.append((1 - F.cosine_similarity(x_p, p_p, dim=0)).item())
                match_prob.append(m_p.item())
                if (query.scene_id == db.scene_id): true_match.append(1)
                else: true_match.append(0)

            if (cfg.train.loss_ablation_m or cfg.eval.eval_only_c):
                cos_sims = np.array(cos_sims)
                true_match = np.array(true_match)
                t1 = time.time()
                sorted_indices = np.argsort(cos_sims)
                sorted_indices = sorted_indices[::-1]
                if timer is not None:
                    timer.text2graph_matching_time.append(time.time() - t1)
                    timer.text2graph_matching_iter.append(1)
                cos_sims = cos_sims[sorted_indices]
                true_match = true_match[sorted_indices]
            elif (cfg.train.loss_ablation_c):
                match_prob = np.array(match_prob)
                true_match = np.array(true_match)
                t1 = time.time()
                sorted_indices = np.argsort(match_prob)
                if timer is not None:
                    timer.text2graph_matching_time.append(time.time() - t1)
                    timer.text2graph_matching_iter.append(1)
                match_prob = match_prob[sorted_indices]
                true_match = true_match[sorted_indices]
            else:
                match_prob = np.array(match_prob)
                true_match = np.array(true_match)
                t1 = time.time()
                sorted_indices = np.argsort(match_prob)
                if timer is not None:
                    timer.text2graph_matching_time.append(time.time() - t1)
                    timer.text2graph_matching_iter.append(1)
                match_prob = match_prob[sorted_indices]
                true_match = true_match[sorted_indices]

            scene_ids_tset = [scene_ids_tset[i] for i in sorted_indices]

            for k in valid_top_k:
                if (1 in true_match[-k:]): valid[k].append(1)
                else: valid[k].append(0)

        for k in valid_top_k:
            if k not in all_valid: all_valid[k] = []
            all_valid[k].append(np.mean(valid[k]))

    accuracy = {k: (np.mean(all_valid[k]), np.std(all_valid[k])) for k in valid_top_k}
    if fold is not None:
        for k in accuracy: wandb.log({f'accuracy_{str(mode)}_top_{k}_fold_{fold}': accuracy[k]})
    else:
        for k in accuracy: wandb.log({f'accuracy_{str(mode)}_top_{k}': accuracy[k]})
    print(f'accuracies: {accuracy}')
    model.train()

    return accuracy


def train_with_cross_val(dataset, database_3dssg, model, folds, epochs, batch_size, entire_training_set, cfg, device='cuda'):
    """Trains the model with optional k-fold cross-validation.

    If ``entire_training_set`` is True, trains on all data without validation.
    Otherwise, splits by scene into k folds and trains/evaluates each fold.

    Args:
        dataset: List of training SceneGraph objects.
        database_3dssg: Dictionary mapping scene IDs to 3DSSG SceneGraph objects.
        model: BigGNN model to train.
        folds: Number of cross-validation folds.
        epochs: Number of training epochs.
        batch_size: Number of graphs per batch.
        entire_training_set: If True, skip cross-validation and train on all data.
        cfg: Hydra DictConfig.

    Returns:
        The trained model.
    """
    ckpt_dir = Path(cfg.paths.checkpoint_dir)
    if entire_training_set:
        if cfg.train.continue_training:
            model = BigGNN(cfg.model.N, cfg.model.heads, cfg.model.embed_dim, cfg.model.dropout).to(device)
            model_dict = torch.load(ckpt_dir / f'{cfg.train.continue_training_model}.pt', weights_only=False)
            model.load_state_dict(model_dict)
        else: model = BigGNN(cfg.model.N, cfg.model.heads, cfg.model.embed_dim, cfg.model.dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

        starting_epoch = 1
        if (cfg.train.continue_training):
            starting_epoch = cfg.train.continue_training
        epochs = epochs + starting_epoch
        for epoch in tqdm(range(starting_epoch, epochs)):
            _ = train(model=model,
                               optimizer=optimizer,
                               database_3dssg=database_3dssg,
                               dataset=dataset,
                               batch_size=batch_size,
                               fold=None,
                               cfg=cfg,
                               device=device)
            if epoch % 2 == 0:
                torch.save(model.state_dict(), ckpt_dir / f'{cfg.train.model_name}_epoch_{epoch}_checkpoint.pt')
        return model

    # K-fold cross-validation
    val_losses, accs, durations = [], [], []
    for fold, (train_idx, val_idx) in enumerate(k_fold_by_scene(dataset, folds)):
        train_dataset = [dataset[i] for i in train_idx]
        val_dataset = [dataset[i] for i in val_idx]

        print(f'length of training set in fold {fold}: {len(train_dataset)}')
        print(f'length of validation set in fold {fold}: {len(val_dataset)}')

        if cfg.train.continue_training:
            model = BigGNN(cfg.model.N, cfg.model.heads, cfg.model.embed_dim, cfg.model.dropout).to(device)
            model_dict = torch.load(ckpt_dir / f'{cfg.train.continue_training_model}.pt', weights_only=False)
            model.load_state_dict(model_dict)
        else: model = BigGNN(cfg.model.N, cfg.model.heads, cfg.model.embed_dim, cfg.model.dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

        starting_epoch = 1
        if (cfg.train.continue_training):
            starting_epoch = cfg.train.continue_training
        epochs = epochs + starting_epoch
        for epoch in tqdm(range(starting_epoch, epochs)):
            _ = train(model=model,
                               optimizer=optimizer,
                               database_3dssg=database_3dssg,
                               dataset=train_dataset,
                               batch_size=batch_size,
                               fold=fold,
                               cfg=cfg,
                               device=device)
            if epoch % 2 == 0:
                torch.save(model.state_dict(), ckpt_dir / f'{cfg.train.model_name}_epoch_{epoch}_checkpoint.pt')
            val_losses.append(eval_loss(model=model,
                                        database_3dssg=database_3dssg,
                                        dataset=val_dataset,
                                        fold=fold,
                                        cfg=cfg,
                                        device=device))
            accs.append(eval_acc(model=model,
                                 database_3dssg=database_3dssg,
                                 dataset=val_dataset,
                                 fold=fold,
                                 cfg=cfg,
                                 eval_iter_count=30,
                                 device=device))
            eval_info = {
                'fold': fold,
                'epoch': epoch,
                'train_loss': _,
                'val_loss': val_losses[-1],
                'val_acc_from_train': accs[-1],
            }
            print(f'Evaluation information: {eval_info}')

        if (cfg.train.skip_k_fold): break

    return model


def run_training(cfg: DictConfig) -> None:
    """Main training entry point.

    Args:
        cfg: Merged Hydra configuration.
    """
    device = cfg.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cpu":
        torch.cuda.empty_cache()
        print(torch.cuda.current_device())
    random.seed(cfg.seed)

    if cfg.train.model_name is None:
        raise ValueError("train.model_name is required. Set via CLI: train.model_name=my_model")
    if (cfg.train.loss_ablation_m and cfg.train.loss_ablation_c):
        raise ValueError("Can only have one loss ablation true at a time")

    ckpt_dir = Path(cfg.paths.checkpoint_dir)

    wandb.init(project="graph2graph",
                mode=cfg.mode,
                config=OmegaConf.to_container(cfg, resolve=True))

    _3dssg_graphs = {}
    _3dssg_scenes = torch.load(cfg.paths.graphs_3dssg, weights_only=False)
    for sceneid in tqdm(_3dssg_scenes):
        _3dssg_graphs[sceneid] = SceneGraph(sceneid,
                                            graph_type='3dssg',
                                            graph=_3dssg_scenes[sceneid],
                                            max_dist=cfg.graph.max_dist,
                                            embedding_type=cfg.graph.embedding_type,
                                            use_attributes=cfg.graph.use_attributes)

    scanscribe_graphs = {}
    scanscribe_scenes = torch.load(cfg.paths.scanscribe_train, weights_only=False)
    for scene_id in tqdm(scanscribe_scenes):
        txtids = scanscribe_scenes[scene_id].keys()
        assert len(set(txtids)) == len(txtids), "Duplicate text IDs found"
        assert len(set(txtids)) == len(range(max([int(id) for id in txtids]) + 1)), "Non-contiguous text IDs"
        for txt_id in txtids:
            txt_id_padded = str(txt_id).zfill(5)
            scanscribe_graphs[scene_id + '_' + txt_id_padded] = SceneGraph(scene_id,
                                                                        txt_id=txt_id,
                                                                        graph_type='scanscribe',
                                                                        graph=scanscribe_scenes[scene_id][txt_id],
                                                                        embedding_type=cfg.graph.embedding_type,
                                                                        use_attributes=cfg.graph.use_attributes)

    print(f'number of scanscribe graphs before removing graphs with 1 edge: {len(scanscribe_graphs)}')
    to_remove = []
    for g in scanscribe_graphs:
        if len(scanscribe_graphs[g].edge_idx[0]) <= 1:
            to_remove.append(g)
    for g in to_remove: del scanscribe_graphs[g]
    print(f'number of scanscribe graphs after removing graphs with 1 edge: {len(scanscribe_graphs)}')
    scanscribe_graphs = list(scanscribe_graphs.values())

    scanscribe_graphs_test = {}
    scanscribe_scenes_test = torch.load(cfg.paths.scanscribe_test, weights_only=False)
    for scene_id in tqdm(scanscribe_scenes_test):
        txtids = scanscribe_scenes_test[scene_id].keys()
        assert len(set(txtids)) == len(txtids), "Duplicate text IDs found"
        assert len(set(txtids)) == len(range(max([int(id) for id in txtids]) + 1)), "Non-contiguous text IDs"
        for txt_id in txtids:
            txt_id_padded = str(txt_id).zfill(5)
            scanscribe_graphs_test[scene_id + '_' + txt_id_padded] = SceneGraph(scene_id,
                                                                        txt_id=txt_id,
                                                                        graph_type='scanscribe',
                                                                        graph=scanscribe_scenes_test[scene_id][txt_id],
                                                                        embedding_type=cfg.graph.embedding_type,
                                                                        use_attributes=cfg.graph.use_attributes)

    print(f'number of scanscribe test graphs before removing: {len(scanscribe_graphs_test)}')
    to_remove = []
    for g in scanscribe_graphs_test:
        if len(scanscribe_graphs_test[g].edge_idx[0]) < 1:
            to_remove.append(g)
    for g in to_remove: del scanscribe_graphs_test[g]
    print(f'number of scanscribe test graphs after removing: {len(scanscribe_graphs_test)}')

    h_graphs_test = torch.load(cfg.paths.human_graphs, weights_only=False)
    h_graphs_remove = [k for k in h_graphs_test if k.split('_')[0] not in _3dssg_graphs]
    print(f'to remove human_graphs, hopefully none: {h_graphs_remove}')
    for k in h_graphs_remove: del h_graphs_test[k]
    assert all([k.split('_')[0] in _3dssg_graphs for k in h_graphs_test]), \
        "All human graph scene IDs must exist in 3DSSG"
    human_graphs_test = {k: SceneGraph(k.split('_')[0],
                                   graph_type='human',
                                   graph=h_graphs_test[k],
                                   embedding_type=cfg.graph.embedding_type,
                                   use_attributes=cfg.graph.use_attributes) for k in h_graphs_test}

    ###################### MEMORY SIZE ANALYSIS ######################
    b_n = 0
    b_e = 0
    b_f = 0
    b_n_h = 0
    b_e_h = 0
    b_f_h = 0
    scanscribe_graphs_list_of_ids = [a.split('_')[0] for a in list(scanscribe_graphs_test.keys())]
    human_graphs_list_of_ids = [a.split('_')[0] for a in list(human_graphs_test.keys())]
    assert all([a in _3dssg_graphs for a in scanscribe_graphs_list_of_ids]), \
        "All ScanScribe test scene IDs must exist in 3DSSG"
    assert all([a in _3dssg_graphs for a in human_graphs_list_of_ids]), \
        "All human test scene IDs must exist in 3DSSG"

    for g in _3dssg_graphs:
        if g in scanscribe_graphs_list_of_ids:
            graph = _3dssg_graphs[g]
            for n in graph.nodes:
                n = graph.nodes[n]
                b_n += np.array(n.features).size * np.array(n.features).itemsize
            b_e += np.array(graph.edge_idx).size * np.array(graph.edge_idx).itemsize
            b_f += np.array(graph.edge_features).size * np.array(graph.edge_features).itemsize
        if g in human_graphs_list_of_ids:
            graph = _3dssg_graphs[g]
            for n in graph.nodes:
                n = graph.nodes[n]
                b_n_h += np.array(n.features).size * np.array(n.features).itemsize
            b_e_h += np.array(graph.edge_idx).size * np.array(graph.edge_idx).itemsize
            b_f_h += np.array(graph.edge_features).size * np.array(graph.edge_features).itemsize

    print(f'SCANSCRIBE b_n: {b_n}, b_e: {b_e}, b_f: {b_f}, total: {b_n + b_e + b_f}')
    print(f'HUMAN b_n_h: {b_n_h}, b_e_h: {b_e_h}, b_f_h: {b_f_h}, total: {b_n_h + b_e_h + b_f_h}')

    if cfg.train.training_with_cross_val:
        if cfg.train.continue_training:
            model = BigGNN(cfg.model.N, cfg.model.heads, cfg.model.embed_dim, cfg.model.dropout).to(device)
            model_dict = torch.load(ckpt_dir / f'{cfg.train.continue_training_model}.pt', weights_only=False)
            model.load_state_dict(model_dict)
        else: model = BigGNN(cfg.model.N, cfg.model.heads, cfg.model.embed_dim, cfg.model.dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
        model = train_with_cross_val(database_3dssg=_3dssg_graphs,
                                        dataset=scanscribe_graphs,
                                        model=model,
                                        folds=cfg.train.folds,
                                        epochs=cfg.train.epoch,
                                        batch_size=cfg.train.batch_size,
                                        entire_training_set=cfg.train.entire_training_set,
                                        cfg=cfg,
                                        device=device)

    ######### SAVE MODEL #########
    model_name = cfg.train.model_name
    cfg_str = OmegaConf.to_yaml(cfg)
    with open(ckpt_dir / f'{model_name}_args.txt', 'w') as f: f.write(cfg_str)
    torch.save(model.state_dict(), ckpt_dir / f'{model_name}.pt')

    t_start = time.perf_counter()
    scanscribe_test_accuracy = eval_acc(model=model,
                                     database_3dssg=_3dssg_graphs,
                                     dataset=list(scanscribe_graphs_test.values()),
                                     fold=None,
                                     cfg=cfg,
                                     mode='scanscribe_test',
                                     device=device)
    human_test_accuracy = eval_acc(model=model,
                                     database_3dssg=_3dssg_graphs,
                                     dataset=list(human_graphs_test.values()),
                                     fold=None,
                                     cfg=cfg,
                                     mode='human_test',
                                     device=device)
    t_end = time.perf_counter()
    print(f'Time elapsed in minutes: {(t_end - t_start) / 60}')

    print(f'Final test set accuracies: scanscribe {scanscribe_test_accuracy}, human {human_test_accuracy}')


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra CLI entry point for training."""
    run_training(cfg)


if __name__ == '__main__':
    main()
