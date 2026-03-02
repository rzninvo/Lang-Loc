"""Evaluation script for trained BigGNN models on ScanScribe, human, and ScanNet test sets."""

import time
from pathlib import Path
import torch
from tqdm import tqdm
import wandb
import random

import hydra
from omegaconf import DictConfig, OmegaConf

from src.data_processing.scene_graph import SceneGraph
from src.models.model_graph2graph import BigGNN
from src.models.train import eval_acc as eval_fn
from src.models.train import format_to_latex
from src.models.timing import Timer


def run_eval(cfg: DictConfig) -> None:
    """Main evaluation entry point.

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

    if cfg.eval.model_name is None:
        raise ValueError("eval.model_name is required. Set via CLI: eval.model_name=my_model")

    ckpt_dir = Path(cfg.paths.checkpoint_dir)
    eval_output_dir = Path(cfg.paths.eval_output_dir)
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    wandb.init(project="graph2graph",
                mode=cfg.mode,
                config=OmegaConf.to_container(cfg, resolve=True))

    # 3DSSG
    _3dssg_graphs = {}
    _3dssg_scenes = torch.load(cfg.paths.graphs_3dssg, weights_only=False)
    for sceneid in tqdm(_3dssg_scenes):
        _3dssg_graphs[sceneid] = SceneGraph(sceneid,
                                            graph_type='3dssg',
                                            graph=_3dssg_scenes[sceneid],
                                            max_dist=cfg.graph.max_dist,
                                            embedding_type=cfg.graph.embedding_type,
                                            use_attributes=cfg.graph.use_attributes)

    # ScanScribe Test
    scanscribe_graphs_test = {}
    scanscribe_scenes = torch.load(cfg.paths.scanscribe_text, weights_only=False)
    for scene_id in tqdm(scanscribe_scenes):
        scanscribe_graphs_test[scene_id] = SceneGraph(scene_id,
                                                txt_id=None,
                                                graph_type='human',
                                                graph=scanscribe_scenes[scene_id],
                                                embedding_type=cfg.graph.embedding_type,
                                                use_attributes=cfg.graph.use_attributes)

    print(f'number of scanscribe test graphs before removing: {len(scanscribe_graphs_test)}')
    to_remove = []
    for g in scanscribe_graphs_test:
        if len(scanscribe_graphs_test[g].edge_idx[0]) < 1:
            to_remove.append(g)
    for g in to_remove: del scanscribe_graphs_test[g]
    print(f'number of scanscribe test graphs after removing: {len(scanscribe_graphs_test)}')

    # Human Test
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

    scannet_test_graphs = torch.load(cfg.paths.sgfusion_graphs, weights_only=False)
    scannet_test_graphs = {k: SceneGraph(k,
                                      graph_type='human',
                                      graph=scannet_test_graphs[k],
                                      embedding_type=cfg.graph.embedding_type,
                                      use_attributes=cfg.graph.use_attributes) for k in scannet_test_graphs}

    scannet_test_text_graphs = torch.load(cfg.paths.sgfusion_text_graphs, weights_only=False)
    scannet_test_text_graphs = {k: SceneGraph(k,
                                        graph_type='human',
                                        graph=scannet_test_text_graphs[k],
                                        embedding_type=cfg.graph.embedding_type,
                                        use_attributes=cfg.graph.use_attributes) for k in scannet_test_text_graphs}

    model_name = cfg.eval.model_name
    model_state_dict = torch.load(ckpt_dir / f'{model_name}.pt', weights_only=False)
    model = BigGNN(cfg.model.N, cfg.model.heads, cfg.model.embed_dim, cfg.model.dropout).to(device)
    model.load_state_dict(model_state_dict)

    if cfg.eval.eval_entire_dataset:
        model_name = model_name + '_topkoutofentire_'
    if cfg.eval.eval_only_c:
        model_name = model_name + '_eval_only_c'
    if cfg.eval.scannet:
        model_name = model_name + '_scannet_'
    if cfg.eval.scanscribe_auto_gen:
        model_name = model_name + '_scanscribe_auto_gen_'
    model_name = model_name + '_' + str(cfg.eval.eval_iters)

    start = time.time()
    scanscribe_timer = Timer()
    scanscribe_test_acc = eval_fn(model=model,
                                    database_3dssg=_3dssg_graphs,
                                    dataset=list(scanscribe_graphs_test.values()),
                                    fold=None,
                                    cfg=cfg,
                                    mode='scanscribe_test',
                                    timer=scanscribe_timer,
                                    device=device)
    print(f'accuracy on scanscribe test set: {scanscribe_test_acc}')
    end_scanscribe = time.time()
    print(f'time for scanscribe test set: {end_scanscribe - start}')
    with open(eval_output_dir / f'{model_name}_scanscribe_test_acc.txt', 'w') as f:
        scanscribe_test_acc = format_to_latex(scanscribe_test_acc)
        f.write(f'{scanscribe_test_acc}')

    start = time.time()
    human_timer = Timer()
    human_test_acc = eval_fn(model=model,
                                    database_3dssg=_3dssg_graphs,
                                    dataset=list(human_graphs_test.values()),
                                    fold=None,
                                    cfg=cfg,
                                    mode='human_test',
                                    valid_top_k=list(cfg.eval.valid_top_k),
                                    timer=human_timer,
                                    device=device)
    print(f'accuracy on human test set: {human_test_acc}')
    end_human = time.time()
    print(f'time for human test set: {end_human - start}')
    with open(eval_output_dir / f'{model_name}_human_test_acc.txt', 'w') as f:
        human_test_acc = format_to_latex(human_test_acc)
        f.write(f'{human_test_acc}')


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra CLI entry point for evaluation."""
    run_eval(cfg)


if __name__ == "__main__":
    main()
