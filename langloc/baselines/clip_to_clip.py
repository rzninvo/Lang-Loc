"""
CLIP-to-CLIP baseline for coarse localization.

Computes cosine similarity between CLIP text embeddings (from scene descriptions)
and CLIP image embeddings (from 3RScan frames) to perform text-to-image or
image-to-text retrieval.

Ported from upstream research code; hardcoded paths replaced with argparse arguments.
"""

import torch
import clip
from PIL import Image
import json
import re
import os
import random
from tqdm import tqdm
import numpy as np
import multiprocessing as mp
import argparse
import time


# ---------------------------------------------------------------------------
# Timer (inlined from timing_util.py)
# ---------------------------------------------------------------------------

class Timer:
    def __init__(self):
        self.start_time = time.time()
        self.total_time = 0
        self.clip2clip_text_embedding_time = []
        self.clip2clip_text_embedding_iter = []

        self.clip2clip_matching_score_time = []
        self.clip2clip_matching_score_iter = []
        # could be combined with above for a total time
        self.clip2clip_matching_time = []
        self.clip2clip_matching_iter = []

    def save(self, path, args):
        with open(path, 'w') as f:
            assert len(self.clip2clip_text_embedding_time) == len(self.clip2clip_text_embedding_iter)
            assert len(self.clip2clip_matching_score_time) == len(self.clip2clip_matching_score_iter)
            assert len(self.clip2clip_matching_time) == len(self.clip2clip_matching_iter)
            assert sum(self.clip2clip_matching_iter) == args.eval_iter * args.eval_iter_count

            f.write(f'start_time: {self.start_time}\n')
            f.write(f'total_time: {self.total_time}\n')
            f.write(f'clip2clip_text_embedding_time: {sum(self.clip2clip_text_embedding_time)}\n')
            f.write(f'clip2clip_text_embedding_iter: {sum(self.clip2clip_text_embedding_iter)}\n')
            f.write(f'clip2clip_matching_score_time: {sum(self.clip2clip_matching_score_time)}\n')
            f.write(f'clip2clip_matching_score_iter: {sum(self.clip2clip_matching_score_iter)}\n')
            f.write(f'clip2clip_matching_time: {sum(self.clip2clip_matching_time)}\n')
            f.write(f'clip2clip_matching_iter: {sum(self.clip2clip_matching_iter)}\n')

            time_for_embedding = sum(self.clip2clip_text_embedding_time) / sum(self.clip2clip_text_embedding_iter)
            time_for_matching_score = sum(self.clip2clip_matching_score_time) / sum(self.clip2clip_matching_score_iter)
            time_for_matching = sum(self.clip2clip_matching_time) / sum(self.clip2clip_matching_iter)

            f.write(f'Embedding time, avg time for 1 encode_text(str): {time_for_embedding}\n')
            f.write(f'Std of embedding time: {np.std(self.clip2clip_text_embedding_time)}\n')
            f.write(f'Matching score time, avg time for 1 matching_score: {time_for_matching_score}\n')
            f.write(f'Std of matching score time: {np.std(self.clip2clip_matching_score_time)}\n')
            f.write(f'Matching time, avg time for 1 matching, or sorting within {args.out_of}: {time_for_matching}\n')
            f.write(f'Std of matching time: {np.std(self.clip2clip_matching_time)}\n')

            calc_time = time_for_embedding + time_for_matching_score * args.out_of + time_for_matching
            f.write(f'Total run time for 1 text matching against {args.out_of} database scenes: {calc_time}\n')


def main():
    # -----------------------------------------------------------------------
    # Argument parsing
    # -----------------------------------------------------------------------

    parser = argparse.ArgumentParser(description='CLIP to CLIP baseline')

    # Paths (replace hardcoded paths from upstream)
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Base directory for cache files and outputs '
                             '(replaces /home/julia/Documents/h_coarse_loc/baselines/CLIP-to-CLIP/)')
    parser.add_argument('--graphs_test_path', type=str, default=None,
                        help='Path to scanscribe graphs test .pt file')
    parser.add_argument('--scanscribe_cleaned_path', type=str, default=None,
                        help='Path to scanscribe_cleaned.json')
    parser.add_argument('--rscan_root', type=str, default=None,
                        help='Path to 3RScan image root directory')
    parser.add_argument('--human_data_path', type=str, default=None,
                        help='Path to human data JSON (line-delimited)')

    # Evaluation parameters
    parser.add_argument('--eval_iter', type=int, default=10000,
                        help='Number of iterations to evaluate')
    parser.add_argument('--top', type=int, default=[1, 2, 3, 5],
                        help='Top-k values for evaluation')
    parser.add_argument('--out_of', type=int, default=10,
                        help='Number of candidates to rank among')
    parser.add_argument('--eval_iter_count', type=int, default=100,
                        help='Number of samples per evaluation iteration')
    parser.add_argument('--dataset', type=str, default='scanscribe',
                        help='scanscribe or human')
    parser.add_argument('--direction', type=str, default='image_to_text',
                        help='text_to_image or image_to_text')
    parser.add_argument('--eval_entire_dataset', action='store_true',
                        help='Evaluate over the entire dataset')
    parser.add_argument('--take_avg_sent_emb_first', action='store_true',
                        help='Average sentence embeddings before matching')
    parser.add_argument('--unsampled', action='store_true',
                        help='Use all images instead of sampled subset')

    args = parser.parse_args()

    scanscribe_timer = Timer()
    human_timer = Timer()

    # -------------------------------------------------------------------
    # Utility functions
    # -------------------------------------------------------------------

    def take_avg_across_scenes(score, all_sentence_scenes):
        """For every sentence, take the average of scores from the same scene."""
        assert len(score) == len(all_sentence_scenes)
        scene_score = {}
        seen_scenes = []
        for i, scene in enumerate(all_sentence_scenes):
            if scene not in seen_scenes:
                seen_scenes.append(scene)
                scene_indices = [j for j, s in enumerate(all_sentence_scenes) if s == scene]
                temp_score = [score[scene_ind] for scene_ind in scene_indices]
                scene_score[scene] = sum(temp_score) / len(temp_score)
        return scene_score

    def sum_over_all_sentence_scenes(time_list, all_sentence_scenes):
        scene_time = {}
        seen_scenes = []
        for i, scene in enumerate(all_sentence_scenes):
            if scene not in seen_scenes:
                seen_scenes.append(scene)
                scene_indices = [j for j, s in enumerate(all_sentence_scenes) if s == scene]
                temp_time = [time_list[scene_ind] for scene_ind in scene_indices]
                scene_time[scene] = sum(temp_time)
        return list(scene_time.values()), [1 for _ in scene_time.keys()]

    def cos_sim(a, b):
        a = np.reshape(a, (512,))
        b = np.reshape(b, (512,))
        t1 = time.time()
        sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        timer.clip2clip_matching_score_time.append(time.time() - t1)
        timer.clip2clip_matching_score_iter.append(1)
        return sim

    def get_average_sentence_embedding(all_sentences_encoded, all_sentence_scenes):
        """Turn all_sentences_encoded into an average of all the phrases in the same sentence."""
        scene_sentence_embedding = {}
        seen_scenes = []
        for i, scene in enumerate(all_sentence_scenes):
            if scene not in seen_scenes:
                seen_scenes.append(scene)
                scene_indices = [j for j, s in enumerate(all_sentence_scenes) if s == scene]
                temp_sentence_embedding = [all_sentences_encoded[scene_ind] for scene_ind in scene_indices]
                scene_sentence_embedding[scene] = np.mean(temp_sentence_embedding, axis=0)
        return list(scene_sentence_embedding.values()), list(scene_sentence_embedding.keys())

    def get_top(index_in_scores, all_max_scores):
        scores = [all_max_scores[i] for i, _ in index_in_scores]
        scores = np.array(scores)
        max_scores_ind = np.argsort(scores)[::-1]
        max_scores = scores[max_scores_ind]
        return random.sample(list(max_scores), 1)[0], index_in_scores[max_scores_ind[0]][1]

    def print_form(acc):
        for k, v in acc.items():
            print(f'{k} top: {v[0] * 100:.2f}% variance: {v[1] * 100:.2f}%')
            print(f'${v[0] * 100:.2f}\\pm{v[1] * 100:.2f}$')

    # -------------------------------------------------------------------
    # CLIP model loading
    # -------------------------------------------------------------------

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    # -------------------------------------------------------------------
    # Dataset loading: ScanScribe
    # -------------------------------------------------------------------

    if args.dataset == "scanscribe":
        scanscribe_graphs_test = torch.load(args.graphs_test_path)
        scanscribe_test_scenes = list(scanscribe_graphs_test.keys())

        scanscribe_cleaned = json.load(open(args.scanscribe_cleaned_path, 'r'))

        scene_sentences_tuples = []
        scene_count = 0
        print('Getting sentences for each scene')
        for scene in tqdm(scanscribe_cleaned):
            if scene in scanscribe_test_scenes:
                scene_count += 1
                for id, sentence in enumerate(scanscribe_cleaned[scene]):
                    sentence = re.split(r'[.,]', sentence)
                    sentence = [s.strip() for s in sentence]
                    sentence = [s for s in sentence if len(s) > 0]
                    scene_sentences_tuples.append((scene + '_' + str(id), sentence))
        print(f'number of sentence descriptions in scanscribe dataset: {len(scene_sentences_tuples)}')
        assert scene_count == 55

        sample_count = 100
        scene_images_tuples = []
        print('Getting images for each scene scanscribe')

        cache_path = os.path.join(args.data_dir, f'scene_images_tuples{"_unsampled" if args.unsampled else ""}.pt')
        if os.path.exists(cache_path):
            scene_images_tuples = torch.load(cache_path)
        else:
            for scene in tqdm(scanscribe_test_scenes):
                scene_images_folder = os.path.join(args.rscan_root, scene, 'sequence')
                scene_images = os.listdir(scene_images_folder)
                scene_images = [os.path.join(scene_images_folder, image) for image in scene_images if image.endswith('.jpg')]
                scene_images_encoded = []
                for img in scene_images:
                    scene_images_encoded.append(model.encode_image(preprocess(Image.open(img)).unsqueeze(0).to(device)).detach().cpu().numpy())
                torch.cuda.empty_cache()
                scene_images_tuples.append((scene_images, scene_images_encoded))
            torch.save(scene_images_tuples, cache_path)

        all_sentences = [sentence for scene, sentences in scene_sentences_tuples for sentence in sentences]
        all_sentence_scenes = [scene for scene, sentences in scene_sentences_tuples for sentence in sentences]
        assert len(all_sentences) == len(all_sentence_scenes)
        print(len(scene_sentences_tuples))

    # -------------------------------------------------------------------
    # Dataset loading: Human
    # -------------------------------------------------------------------

    if args.dataset == "human":
        human_data = open(args.human_data_path, 'r').readlines()
        human_data = [json.loads(line) for line in human_data]
        print(human_data[0])

        scene_sentences_tuples_human = []
        print('Getting sentences for each scene, human')
        scenes = []
        human_scenes = []
        for idx, text_data in enumerate(human_data):
            text_data['scanId'] = text_data['scanId'].split('.')[0] + '_' + str(idx)
            scenes.append(text_data['scanId'])
            human_scenes.append(text_data['scanId'].split('/')[0])

            sentence = text_data['description']
            sentence = re.split(r'[.,]', sentence)
            sentence = [s.strip() for s in sentence]
            sentence = [s for s in sentence if len(s) > 0]
            scene_sentences_tuples_human.append((text_data['scanId'], sentence))
        all_sentences_human = [sentence for scene, sentences in scene_sentences_tuples_human for sentence in sentences]
        all_sentence_scenes_human = [scene for scene, sentences in scene_sentences_tuples_human for sentence in sentences]
        assert len(all_sentences_human) == len(all_sentence_scenes_human)
        assert len(scenes) == len(set(scenes))

    # -------------------------------------------------------------------
    # Encode sentences
    # -------------------------------------------------------------------

    if args.dataset == "human":
        print('encoding human sentences')
        all_sentences_encoded_human = []
        cache_path = os.path.join(args.data_dir, 'all_sentences_encoded_human.pt')
        all_sentences_encoded_human = torch.load(cache_path)

        # Human images encoding
        sample_count = 100
        scene_images_tuples_human = []
        print('Getting images for each scene')
        human_cache_path = os.path.join(args.data_dir, f'scene_images_tuples_human{"_unsampled" if args.unsampled else ""}.pt')
        if os.path.exists(human_cache_path):
            scene_images_tuples_human = torch.load(human_cache_path)
        else:
            for scene in tqdm(human_scenes):
                scene_images_folder = os.path.join(args.rscan_root, scene, 'sequence')
                scene_images = os.listdir(scene_images_folder)
                scene_images = [os.path.join(scene_images_folder, image) for image in scene_images if image.endswith('.jpg')]
                scene_images_encoded = []
                for img in scene_images:
                    scene_images_encoded.append(model.encode_image(preprocess(Image.open(img)).unsqueeze(0).to(device)).detach().cpu().numpy())
                torch.cuda.empty_cache()
                scene_images_tuples_human.append((scene_images, scene_images_encoded))
            torch.save(scene_images_tuples_human, human_cache_path)

    if args.dataset == "scanscribe":
        print('encoding scanscribe sentences')
        all_sentences_encoded = []
        cache_path = os.path.join(args.data_dir, 'all_sentences_encoded.pt')
        all_sentences_encoded = torch.load(cache_path)
        assert len(all_sentences_encoded) == len(all_sentences)
        print(f'len of all_sentences_encoded after encoding: {len(all_sentences_encoded)}')

    # -------------------------------------------------------------------
    # Converge datasets into common variables
    # -------------------------------------------------------------------

    dataset = args.dataset
    timer = scanscribe_timer
    if dataset == "human":
        all_sentences_encoded = all_sentences_encoded_human
        all_sentence_scenes = all_sentence_scenes_human
        scene_sentences_tuples = scene_sentences_tuples_human
        all_sentences = all_sentences_human
        folder_name = 'image_best_desc_human'
        max_scores_per_scene_folder_name = 'max_scores_per_scene_human'
        timer = human_timer
        scene_images_tuples = scene_images_tuples_human
    elif dataset == "scanscribe":
        folder_name = 'image_best_desc'
        max_scores_per_scene_folder_name = 'max_scores_per_scene'
        timer = scanscribe_timer
    else:
        print("please enter dataset name")
        exit()

    if args.unsampled:
        folder_name += '_unsampled'
        max_scores_per_scene_folder_name += '_unsampled'

    folder_path = os.path.join(args.data_dir, folder_name)
    max_scores_path = os.path.join(args.data_dir, max_scores_per_scene_folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    if not os.path.exists(max_scores_path):
        os.makedirs(max_scores_path)

    # -------------------------------------------------------------------
    # Averaging sentence embeddings (optional)
    # -------------------------------------------------------------------

    if args.take_avg_sent_emb_first:
        print('averaging sentence embeddings FIRST')
        all_sentences_encoded, all_sentence_scenes = get_average_sentence_embedding(all_sentences_encoded, all_sentence_scenes)
        assert len(all_sentences_encoded) == len(all_sentence_scenes)
        print(f'len of all_sentences_encoded after averaging: {len(all_sentences_encoded)}')
        print(f'first few of all_sentence_scenes: {all_sentence_scenes[:5]}')

    # -------------------------------------------------------------------
    # Scoring functions
    # -------------------------------------------------------------------

    def f_avg_sent_emb_first(tuple_pair):
        scene_ids_img, img_encoded = tuple_pair
        scene_id = scene_ids_img[0].split('/')[-3]

        sentence_to_best_img_score = {}
        for sent_idx, sent_emb in enumerate(all_sentences_encoded):
            cos_sims = []
            sent_emb = np.array(sent_emb)
            for idx, image in enumerate(img_encoded):
                scene_id_img = scene_ids_img[idx]
                cos_sims.append(cos_sim(image, sent_emb))
            assert len(cos_sims) == len(img_encoded)
            sentence_to_best_img_score[all_sentence_scenes[sent_idx]] = max(cos_sims)
        return sentence_to_best_img_score

    @torch.no_grad()
    def f(tuple_pair):
        scene_ids_img, img_encoded = tuple_pair

        for idx, image in enumerate(img_encoded):
            scene_id_img = scene_ids_img[idx]
            with torch.no_grad():
                cos_sims = []
                for s in all_sentences_encoded:
                    cos_sims.append(cos_sim(image, s))
                cos_sims_by_scene_desc = take_avg_across_scenes(cos_sims, all_sentence_scenes)
                assert len(cos_sims_by_scene_desc) == len(scene_sentences_tuples)

            scene_sub = scene_id_img.split("/")[-3]
            scene_sub_dir = os.path.join(folder_path, scene_sub)
            if not os.path.exists(scene_sub_dir):
                os.makedirs(scene_sub_dir)
            fname = os.path.join(scene_sub, scene_id_img.split("/")[-1])
            torch.save(cos_sims_by_scene_desc, os.path.join(folder_path, fname + '.pt'))

    def get_one_img_scene_to_desc_scene(folder):
        prefix = os.path.join(folder_path, '')
        files = os.listdir(prefix + folder)
        files = [os.path.join(prefix + folder, file) for file in files]
        scores = [torch.load(file) for file in files]
        m = []
        for score in scores:
            m.append(list(score.values()))
        m = np.array(m)
        max_scores = m.max(axis=0)
        assert len(max_scores) == len(scores[0])
        scene_max_dir = os.path.join(max_scores_path, folder)
        if not os.path.exists(scene_max_dir):
            os.makedirs(scene_max_dir)
        torch.save(max_scores, os.path.join(scene_max_dir, 'max_scores.pt'))

    # -------------------------------------------------------------------
    # Evaluation with averaged sentence embeddings
    # -------------------------------------------------------------------

    def get_all_scores_sent_emb_first(image_to_text_score_mapping, desc_scene_ids):
        img_scene_ids = list(image_to_text_score_mapping.keys())
        img_scene_ids_idx = {scene: i for i, scene in enumerate(img_scene_ids)}

        if args.direction == "text_to_image":
            in_top_w_var = {k: [] for k in args.top}
            for _ in range(args.eval_iter):
                in_top = {k: [] for k in args.top}
                for desc_i, desc_scene_id in enumerate(random.sample(list(desc_scene_ids), args.eval_iter_count)):

                    if dataset == "scanscribe":
                        scene_id = desc_scene_id.split('_')[0]
                    elif dataset == "human":
                        scene_id = desc_scene_id.split('/')[0]

                    removed = img_scene_ids.copy()
                    removed.remove(scene_id)
                    sample_img_ids = random.sample(removed, args.out_of - 1)
                    sample_img_ids.append(scene_id)
                    assertion_sample_img_ids = sample_img_ids.copy()
                    assertion_sample_img_ids.append(scene_id)
                    sample_img_ids_idx = [img_scene_ids_idx[im_id] for im_id in assertion_sample_img_ids]
                    assert len(set(assertion_sample_img_ids)) == args.out_of

                    scores = [image_to_text_score_mapping[img_id][desc_scene_id] for img_id in sample_img_ids]
                    scores = np.array(scores)
                    t1 = time.time()
                    max_scores_ind = np.argsort(scores)[::-1]
                    timer.clip2clip_matching_time.append(time.time() - t1)
                    timer.clip2clip_matching_iter.append(1)
                    max_scores = scores[max_scores_ind]
                    for k in in_top:
                        in_top[k].append(args.out_of - 1 in max_scores_ind[:k])

                assert len(list(in_top.values())[0]) == args.eval_iter_count
                for k in in_top_w_var:
                    in_top_w_var[k].append(sum(in_top[k]) / len(in_top[k]))

            in_top_w_var = {k: (sum(in_top_w_var[k]) / len(in_top_w_var[k]), np.var(in_top_w_var[k])) for k in in_top_w_var}
            print_form(in_top_w_var)

        elif args.direction == "image_to_text":
            desc_scene_ids_by_scene = {}
            for i, scene in enumerate(desc_scene_ids):
                if dataset == "scanscribe":
                    scene_id = scene.split('_')[0]
                elif dataset == "human":
                    scene_id = scene.split('/')[0]
                if scene_id not in desc_scene_ids_by_scene:
                    desc_scene_ids_by_scene[scene_id] = [(i, scene)]
                else:
                    desc_scene_ids_by_scene[scene_id].append((i, scene))
            print(len(desc_scene_ids_by_scene))
            assert len(desc_scene_ids_by_scene) == 55 or len(desc_scene_ids_by_scene) == 142
            assert len(list(desc_scene_ids_by_scene.keys())) == len(img_scene_ids)

            in_top_w_var = {k: [] for k in args.top}
            for _ in range(args.eval_iter):
                in_top = {k: [] for k in args.top}
                for img_i, img_scene_id in enumerate(random.sample(img_scene_ids, args.eval_iter_count)):

                    removed = list(desc_scene_ids_by_scene.keys())
                    removed.remove(img_scene_id)
                    sample_desc_ids = random.sample(removed, args.out_of - 1)
                    sample_desc_ids.append(img_scene_id)

                    assertion_sample_desc_ids = sample_desc_ids.copy()
                    assertion_sample_desc_ids.append(img_scene_id)
                    assert len(set(assertion_sample_desc_ids)) == args.out_of

                    sampled_tuple = [random.sample(desc_scene_ids_by_scene[sample_desc_id], 1)[0] for sample_desc_id in sample_desc_ids]
                    scores = [image_to_text_score_mapping[img_scene_id][desc_scene_id] for _, desc_scene_id in sampled_tuple]
                    assert len(scores) == args.out_of
                    scores = np.array(scores)
                    t1 = time.time()
                    max_scores_ind = np.argsort(scores)[::-1]
                    timer.clip2clip_matching_time.append(time.time() - t1)
                    timer.clip2clip_matching_iter.append(1)
                    max_scores = scores[max_scores_ind]

                    for k in in_top:
                        in_top[k].append(args.out_of - 1 in max_scores_ind[:k])

                assert len(list(in_top.values())[0]) == args.eval_iter_count
                for k in in_top_w_var:
                    in_top_w_var[k].append(sum(in_top[k]) / len(in_top[k]))

            in_top_w_var = {k: (sum(in_top_w_var[k]) / len(in_top_w_var[k]), np.var(in_top_w_var[k])) for k in in_top_w_var}
            print_form(in_top_w_var)

    # -------------------------------------------------------------------
    # Evaluation with per-image scores
    # -------------------------------------------------------------------

    def get_all_scores(scene_names, text_desc_ids):

        if dataset == "human":
            scene_names = set([scene.split('/')[0] for scene in text_desc_ids])

        all_max_scores = []
        for scene in scene_names:
            max_score = torch.load(os.path.join(max_scores_path, scene, 'max_scores.pt'))
            all_max_scores.append(max_score)
        all_max_scores = np.array(all_max_scores)  # Num scenes X Num descriptions

        # Quick stats on a random line
        line = random.choice(all_max_scores)
        print(np.mean(line))
        print(np.var(line))
        print(np.max(line))
        print(np.min(line))

        print(np.mean(all_max_scores, axis=0)[:5])
        print(np.var(all_max_scores, axis=0)[:5])
        print(np.max(all_max_scores, axis=0)[:5])
        print(np.min(all_max_scores, axis=0)[:5])

        img_scene_ids = scene_names
        desc_scene_ids = text_desc_ids

        assert len(img_scene_ids) == 55 or len(img_scene_ids) == 142
        assert len(desc_scene_ids) == 1116 or len(desc_scene_ids) == 147
        assert all_max_scores.shape == (55, 1116) or all_max_scores.shape == (142, 147)

        desc_scene_ids_by_scene = {}
        for i, scene in enumerate(desc_scene_ids):
            if dataset == "scanscribe":
                scene_id = scene.split('_')[0]
            elif dataset == "human":
                scene_id = scene.split('/')[0]
            if scene_id not in desc_scene_ids_by_scene:
                desc_scene_ids_by_scene[scene_id] = [(i, scene)]
            else:
                desc_scene_ids_by_scene[scene_id].append((i, scene))
        print(len(desc_scene_ids_by_scene))
        assert len(desc_scene_ids_by_scene) == 55 or len(desc_scene_ids_by_scene) == 142
        assert len(list(desc_scene_ids_by_scene.keys())) == len(img_scene_ids)

        img_scene_ids_idx = {scene: i for i, scene in enumerate(img_scene_ids)}

        # Matching 1 text to N images
        if args.direction == "text_to_image":
            in_top_w_var = {k: [] for k in args.top}
            for _ in range(args.eval_iter):
                in_top = {k: [] for k in args.top}
                for desc_i, desc_scene_id in enumerate(random.sample(desc_scene_ids, args.eval_iter_count)):

                    if dataset == "scanscribe":
                        scene_id = desc_scene_id.split('_')[0]
                    elif dataset == "human":
                        scene_id = desc_scene_id.split('/')[0]

                    removed = list(img_scene_ids)
                    removed.remove(scene_id)
                    sample_img_ids = random.sample(removed, args.out_of - 1)
                    assertion_sample_img_ids = sample_img_ids.copy()
                    assertion_sample_img_ids.append(scene_id)
                    sample_img_ids_idx = [img_scene_ids_idx[im_id] for im_id in assertion_sample_img_ids]
                    assert len(set(assertion_sample_img_ids)) == args.out_of

                    scores = [all_max_scores[img_idx, desc_i] for img_idx in sample_img_ids_idx]
                    scores = np.array(scores)
                    t1 = time.time()
                    max_scores_ind = np.argsort(scores)[::-1]
                    timer.clip2clip_matching_time.append(time.time() - t1)
                    timer.clip2clip_matching_iter.append(1)
                    max_scores = scores[max_scores_ind]

                    for k in in_top:
                        in_top[k].append(args.out_of - 1 in max_scores_ind[:k])

                assert len(list(in_top.values())[0]) == args.eval_iter_count
                for k in in_top_w_var:
                    in_top_w_var[k].append(sum(in_top[k]) / len(in_top[k]))

            in_top_w_var = {k: (sum(in_top_w_var[k]) / len(in_top_w_var[k]), np.var(in_top_w_var[k])) for k in in_top_w_var}
            print_form(in_top_w_var)

        # Matching 1 image to N texts
        elif args.direction == "image_to_text":
            in_top_w_var = {k: [] for k in args.top}
            for _ in range(args.eval_iter):
                in_top = {k: [] for k in args.top}
                for img_i, img_scene_id in enumerate(random.sample(img_scene_ids, args.eval_iter_count)):

                    removed = list(desc_scene_ids_by_scene.keys())
                    removed.remove(img_scene_id)
                    sample_desc_ids = random.sample(removed, args.out_of - 1)
                    assertion_sample_desc_ids = sample_desc_ids.copy()
                    assertion_sample_desc_ids.append(img_scene_id)
                    assert len(set(assertion_sample_desc_ids)) == args.out_of

                    top_match_score, top_match_text_id = get_top(desc_scene_ids_by_scene[img_scene_id], all_max_scores[img_i])
                    sampled_tuple = [random.sample(desc_scene_ids_by_scene[sample_desc_id], 1)[0] for sample_desc_id in sample_desc_ids]
                    scores = [all_max_scores[img_i, desc_idx] for desc_idx, _ in sampled_tuple]
                    scores.append(top_match_score)
                    assert len(scores) == args.out_of
                    scores = np.array(scores)
                    t1 = time.time()
                    max_scores_ind = np.argsort(scores)[::-1]
                    timer.clip2clip_matching_time.append(time.time() - t1)
                    timer.clip2clip_matching_iter.append(1)
                    max_scores = scores[max_scores_ind]

                    for k in in_top:
                        in_top[k].append(args.out_of - 1 in max_scores_ind[:k])

                assert len(list(in_top.values())[0]) == args.eval_iter_count
                for k in in_top_w_var:
                    in_top_w_var[k].append(sum(in_top[k]) / len(in_top[k]))

            in_top_w_var = {k: (sum(in_top_w_var[k]) / len(in_top_w_var[k]), np.var(in_top_w_var[k])) for k in in_top_w_var}
            print_form(in_top_w_var)

    # -------------------------------------------------------------------
    # Main execution
    # -------------------------------------------------------------------

    image_to_text_score_mapping = {}
    if args.take_avg_sent_emb_first:
        mapping_cache = os.path.join(args.data_dir, f'image_to_text_score_mapping_{args.dataset}.pt')
        if os.path.exists(mapping_cache):
            image_to_text_score_mapping = torch.load(mapping_cache)
        else:
            for scene_image in tqdm(scene_images_tuples):
                scene_id = scene_image[0][0].split('/')[-3]
                assert all([scene_id in s for s in scene_image[0]])
                text_score_for_one_scene = f_avg_sent_emb_first(scene_image)
                image_to_text_score_mapping[scene_id] = text_score_for_one_scene
            torch.save(image_to_text_score_mapping, mapping_cache)
    else:
        for scene_images in tqdm(scene_images_tuples):
            f(scene_images)

    print(f'len of image_to_text_score_mapping: {len(image_to_text_score_mapping)}')

    scene_names = os.listdir(folder_path)

    def check(scene_names):
        for s in scene_names:
            prefix = os.path.join(folder_path, '')
            files = os.listdir(prefix + s)
            files = [os.path.join(prefix + s, file) for file in files]
            scores = [torch.load(file) for file in files]
            keys = [list(score.keys()) for score in scores]
            assert all(keys[0] == key for key in keys)
            assert len(keys[0]) == 1116

    scene_names = os.listdir(max_scores_path)
    if args.dataset == "human":
        example_with_text_desc_ids = os.path.join(folder_path, '0ad2d382-79e2-2212-98b3-641bf9d552c1', 'frame-000000.color.jpg.pt')
    elif args.dataset == "scanscribe":
        example_with_text_desc_ids = os.path.join(folder_path, '0ad2d38f-79e2-2212-98d2-9b5060e5e9b5', 'frame-000002.color.jpg.pt')
    else:
        print("please enter dataset name")
        exit()

    if args.eval_entire_dataset:
        args.out_of = len(image_to_text_score_mapping)
        args.top = [1, 5, 10, 20, 30]
        if args.dataset == "human":
            args.top.extend([50, 75])

    text_desc_ids = list(torch.load(example_with_text_desc_ids).keys())
    if args.take_avg_sent_emb_first:
        get_all_scores_sent_emb_first(image_to_text_score_mapping, torch.load(example_with_text_desc_ids).keys())
    else:
        get_all_scores(scene_names, text_desc_ids)


if __name__ == "__main__":
    main()
