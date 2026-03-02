"""Text embedding creation using OpenAI Ada, spaCy word2vec, and CLIP backends."""

from typing import List

import openai
import numpy as np
import torch
import json
import os
import tiktoken
import tqdm

from langloc.utils.utils import load_text_dataset, _get_nlp


_clip_tokenizer = None
_clip_model = None


def _get_clip():
    """Returns the lazily-loaded CLIP tokenizer and model."""
    global _clip_tokenizer, _clip_model
    if _clip_tokenizer is None:
        from transformers import CLIPTokenizer, CLIPModel
        _clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True)
    return _clip_tokenizer, _clip_model


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string.

    Args:
        string: Text to tokenize.
        encoding_name: Name of the tiktoken encoding (e.g. ``'cl100k_base'``).

    Returns:
        Token count.
    """
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def num_tokens_from_list_of_strings(list_of_strings: List[str], encoding_name: str) -> int:
    """Returns the total number of tokens across a list of text strings.

    Args:
        list_of_strings: Texts to tokenize.
        encoding_name: Name of the tiktoken encoding.

    Returns:
        Total token count.
    """
    num_tokens = 0
    for string in list_of_strings:
        num_tokens += num_tokens_from_string(string, encoding_name)
    return num_tokens


def num_tokens_from_dict(dict_of_texts: dict, encoding_name: str) -> int:
    """Returns the total number of tokens across all texts in a dictionary.

    Args:
        dict_of_texts: Mapping from scan IDs to lists of text strings.
        encoding_name: Name of the tiktoken encoding.

    Returns:
        Total token count.
    """
    num_tokens = 0
    for scan_id in dict_of_texts:
        num_tokens += num_tokens_from_list_of_strings(dict_of_texts[scan_id], encoding_name)
    return num_tokens


def check_tokens():
    """Prints the total token count for the default text dataset."""
    scan_ids, dict_of_texts = load_text_dataset()
    num_tokens = num_tokens_from_dict(dict_of_texts, 'cl100k_base')
    print(num_tokens)


def create_embedding_clip(text: str) -> torch.Tensor:
    """Returns a 512-dimensional CLIP text embedding for the given string.

    Args:
        text: Input text to embed.

    Returns:
        A 512-dim tensor.
    """
    tokenizer, model = _get_clip()
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt"
    )
    with torch.no_grad():
        outputs = model.get_text_features(**inputs)
        embedding = outputs.squeeze(0)
    return embedding


def create_embeddings_clip_batch(texts: list[str]) -> torch.Tensor:
    """Returns an (N x 512) tensor of CLIP embeddings for a list of N strings.

    Args:
        texts: List of input texts to embed.

    Returns:
        An ``(N, 512)`` tensor.
    """
    tokenizer, model = _get_clip()
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt"
    )
    with torch.no_grad():
        outputs = model.get_text_features(**inputs)
    return outputs


def create_embedding(text):
    """Creates an Ada-002 embedding via the OpenAI API.

    Args:
        text: Input text to embed.

    Returns:
        List of floats (1536-dim embedding).
    """
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    embedding = response['data'][0]['embedding']
    return embedding


def tokenize_text(filename):
    """Embeds all texts in a dataset file using the Ada API.

    Args:
        filename: Dataset filename to load via ``load_text_dataset``.

    Returns:
        Dictionary mapping scan IDs to lists of Ada embeddings.
    """
    scan_ids, dict_of_texts = load_text_dataset(filename)
    dict_of_embeddings = {}
    for scan_id in tqdm.tqdm(dict_of_texts):
        dict_of_embeddings[scan_id] = []
        for text in dict_of_texts[scan_id]:
            embedding = create_embedding(text)
            dict_of_embeddings[scan_id].append(embedding)
    return dict_of_embeddings


def create_embedding_nlp(text):
    """Creates a 300-dim spaCy word2vec embedding.

    Args:
        text: Input text to embed.

    Returns:
        A 300-dim numpy array.
    """
    doc = _get_nlp()(text)
    embedding = doc.vector
    assert len(embedding) == 300, "Expected 300-dim spaCy vector"
    return embedding


def test_ada_embedding():
    """Prints cosine similarities between sample Ada embeddings for debugging."""
    worda = 'shelf'
    atta = ['brown']
    wordb = 'floor'
    attb = ['tiled']

    emba = create_embedding(worda)
    avg_atta = np.mean([create_embedding(att) for att in atta], axis=0)
    embb = create_embedding(wordb)
    avg_attb = np.mean([create_embedding(att) for att in attb], axis=0)

    print(f'cosine ab word only: {np.dot(emba, embb) / (np.linalg.norm(emba) * np.linalg.norm(embb))}')

    emba = np.add(emba, avg_atta)
    embb = np.add(embb, avg_attb)

    emba_weighted_sum = np.add(emba, 0.2*avg_atta)
    embb_weighted_sum = np.add(embb, 0.2*avg_attb)

    print(f'cosine ab: {np.dot(emba, embb) / (np.linalg.norm(emba) * np.linalg.norm(embb))}')
    print(f'cosine ab weighted sum: {np.dot(emba_weighted_sum, embb_weighted_sum) / (np.linalg.norm(emba_weighted_sum) * np.linalg.norm(embb_weighted_sum))}')


def test_nlp_embedding():
    """Prints cosine similarities between sample spaCy embeddings for debugging."""
    worda = 'shelf'
    wordb = 'bookshelf'
    wordc = 'yellow'
    wordd = 'Jacket'
    emba = create_embedding_nlp(worda)
    embb = np.add(create_embedding_nlp(wordb), create_embedding_nlp(wordc))
    embd = create_embedding_nlp(wordd)
    print(f'cosine ab: {np.dot(emba, embb) / (np.linalg.norm(emba) * np.linalg.norm(embb))}')
    print(f'cosine ad: {np.dot(emba, embd) / (np.linalg.norm(emba) * np.linalg.norm(embd))}')
    print(f'cosine bd: {np.dot(embb, embd) / (np.linalg.norm(embb) * np.linalg.norm(embd))}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_ada', action='store_true', help='Run Ada embedding test')
    parser.add_argument('--filename', type=str, default=None, help='Dataset filename to embed')
    args = parser.parse_args()

    if args.test_ada:
        test_ada_embedding()
    elif args.filename:
        embeddings = tokenize_text(args.filename)
        scripts_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'scripts', 'hugging_face')
        with open(os.path.join(scripts_dir, 'scanscribe_2_embeddings.json'), 'w') as fp:
            json.dump(embeddings, fp)
    else:
        parser.print_help()
