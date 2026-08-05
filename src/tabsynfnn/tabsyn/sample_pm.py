# Portions adapted from amazon-science/TabSyn (Apache-2.0) and modified.
import os
import time
import joblib

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader

from ..util_functions import read_written_hypars
from ..utils_tuning import metric_dict_write
from ..utils.util import set_seed
from ..utils_train import preprocess
from ..prediction_interval_utils import infer_class_labels
from .latent_utils import (
    decode_nearest_response_prototype,
    majority_vote,
    compute_proba,
    regression_average,
)
from .model_pm import MLPDiffusion_pm, Model_pm, sample_pm
from .data_pm import PredictiveModelingDataset


def diffusion_repeated_sample_pm(
        data_path, diffusion_path, sampling_path,
        batch_size=1024, steps=50, num_repeats=50, 
        use_val=False,
        base_seed=None,
        model_filename='model.pt',
    ):
    """
    Repeated sampling, saving each result and majority voting.
    Saves: y_test_{i}.npy for each i, and y_test_majority_vote.npy.
    """    
    print('\nRepeated Sampling begins:\n')

    device = "cuda"
    os.makedirs(sampling_path, exist_ok=True)

    df_hp = read_written_hypars(os.path.join(diffusion_path, 'hyperparameters.txt'))

    y_emb_path = df_hp['y_emb_path']
    emb_alignment_path = df_hp.get('emb_alignment_path', None)
    y_type = df_hp['y_type']
    vae_dat_path = df_hp.get('vae_dat_path', None)
    text_path = df_hp.get('text_path', None)
    image_path = df_hp.get('image_path', None)
    use_tab = df_hp.get('use_tab', False)
    use_text = df_hp.get('use_text', False)
    use_image = df_hp.get('use_image', False)
    text_pooling = df_hp.get('text_pooling', 'mean')
    normalize_tab_cond = df_hp.get('normalize_tab_cond', False)
    normalize_text_tokens = df_hp.get('normalize_text_tokens', False)
    normalize_text_pooled = df_hp.get('normalize_text_pooled', False)
    normalize_projected_cond = df_hp.get('normalize_projected_cond', False)
    alignment_mix_alpha = df_hp.get('alignment_mix_alpha', 1.0)
    raw_vae_dat_path = df_hp.get('raw_vae_dat_path', None)
    raw_text_path = df_hp.get('raw_text_path', None)

    task_type = "multiclass" if y_type == 'categorical' else "regression"
    X_num, X_cat, y, categories_dat, d_numerical, num_inverse, cat_inverse = preprocess(
        data_path, 
        task_type=task_type, 
        is_y_cond=True, 
        inverse=True
    )
    split = 'test' if not use_val else 'val'
    test_dataset = PredictiveModelingDataset(
        y_emb_path, emb_alignment_path, y_type, split,
        vae_dat_path, text_path, image_path,
        use_tab, use_text, use_image,
        alignment_mix_alpha=alignment_mix_alpha,
        raw_vae_dat_path=raw_vae_dat_path,
        raw_text_path=raw_text_path,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    if y_type == 'categorical':
        encoded_y_train = np.asarray(y[0]).reshape(-1).astype(int)
        train_z_y = np.load(os.path.join(y_emb_path, 'train_z_y.npy')).reshape(
            len(encoded_y_train), -1
        )
        class_indices = np.unique(encoded_y_train)
        prototype_array = np.stack(
            [train_z_y[encoded_y_train == value].mean(axis=0) for value in class_indices]
        ).astype(np.float32)
        prototypes = torch.from_numpy(prototype_array).to(device)
    elif y_type == 'continuous':
        scaler = joblib.load(os.path.join(y_emb_path, 'y_scaler.joblib'))

    denoise_fn = MLPDiffusion_pm(
        df_hp['d_in'], df_hp['dim_t'], cond_dim=df_hp['cond_dim'], 
        use_tab=use_tab, use_text=use_text, use_image=use_image,
        text_pooling=text_pooling,
        normalize_tab_cond=normalize_tab_cond,
        normalize_text_tokens=normalize_text_tokens,
        normalize_text_pooled=normalize_text_pooled,
        normalize_projected_cond=normalize_projected_cond,
    ).to(device)
    model = Model_pm(denoise_fn=denoise_fn).to(device)
    model_path = os.path.join(diffusion_path, model_filename)
    print(f"Loading diffusion model: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    if base_seed is None:
        base_seed = np.random.randint(0, 2**32)

    start_time = time.time()

    all_samples = []
    # Per-repeat soft class-probabilities (classification only); averaging these across
    # repeats avoids the per-sample argmax and the 1/M hard-vote quantization.
    all_soft_proba = []
    soft_proba_available = (y_type == 'categorical')

    repeat_seeds = [(base_seed + i) for i in range(num_repeats)]
    for sample_index, seed in enumerate(repeat_seeds):
        print(f"\nSampling repetition {sample_index+1}/{num_repeats}, seed={seed}")
        set_seed(seed)
        save_path = os.path.join(sampling_path, f'y_test_{sample_index}.npy')
        soft_path = os.path.join(sampling_path, f'y_proba_soft_{sample_index}.npy')
        if os.path.exists(save_path):
            # print(f"{save_path} already exists, skipping sampling.")
            print(f"{save_path} already exists, loading saved samples.")
            syn_y = np.load(save_path, allow_pickle=True)
            all_samples.append(syn_y)
            if soft_proba_available:
                if os.path.exists(soft_path):
                    all_soft_proba.append(np.load(soft_path, allow_pickle=True))
                else:
                    print(f"  (no {os.path.basename(soft_path)}; soft-proba averaging disabled)")
                    soft_proba_available = False
            continue

        samples = []
        with torch.no_grad():
            for _, tab_pred, text, img in tqdm(test_loader, desc=f"Sampling {sample_index}"):
                tab_pred = tab_pred.float().to(device)
                text = text.float().to(device)
                img = img.float().to(device)
                x_next = sample_pm(
                    model.denoise_fn_D,
                    num_samples=tab_pred.shape[0],     # batch size
                    dim=df_hp['d_in'],
                    num_steps=steps,
                    tab_pred=tab_pred,
                    text=text,
                    img=img,
                    device=device
                )
                samples.append(x_next.cpu())

        samples = torch.cat(samples, dim=0).to(device)
        
        if y_type == 'categorical':
            syn_y, soft_proba = decode_nearest_response_prototype(
                samples,
                prototypes,
                class_indices,
                X_cat[0].shape[1],
                cat_inverse,
                return_proba=True,
            )
            if soft_proba is not None:
                soft_proba = np.asarray(soft_proba)  # [test_size, n_classes]
                np.save(soft_path, soft_proba)
                all_soft_proba.append(soft_proba)
            else:
                soft_proba_available = False
        elif y_type == 'continuous':
            syn_y_ = samples.cpu().numpy()
            syn_y = scaler.inverse_transform(syn_y_)

        np.save(save_path, syn_y)
        all_samples.append(syn_y)


    # Stack into shape [num_repeats, test_size, 1]
    all_samples_arr = np.stack(all_samples, axis=0)  # [num_repeats, test_size, 1]
    np.save(os.path.join(sampling_path, 'y_test_all.npy'), all_samples_arr[..., 0]) # save as [num_repeats, test_size]

    if y_type == 'categorical':
        # Majority vote across first axis
        maj_vote = majority_vote(all_samples_arr)  # [test_size, 1]
        class_labels = infer_class_labels(y[1].astype(str), y_num_classes=None)
        y_proba = compute_proba(all_samples_arr, class_labels=class_labels)  # [test_size, 1]
        np.save(os.path.join(sampling_path, 'y_test_majority_vote.npy'), maj_vote)
        np.save(os.path.join(sampling_path, 'y_proba.npy'), y_proba)

        # Soft-probability averaging: mean of decoded softmax over repeats.
        if soft_proba_available and len(all_soft_proba) == num_repeats:
            soft_mean = np.stack(all_soft_proba, axis=0).mean(axis=0)  # [test_size, n_classes]
            np.save(os.path.join(sampling_path, 'y_proba_soft_all.npy'), soft_mean)
            if soft_mean.shape[1] == 2:
                # positive-class probability (column 1 == label "1" by decoder order)
                np.save(os.path.join(sampling_path, 'y_proba_soft.npy'), soft_mean[:, 1].reshape(-1, 1))
            else:
                np.save(os.path.join(sampling_path, 'y_proba_soft.npy'), soft_mean)
            print(f"Saved soft-averaged probabilities (y_proba_soft.npy), shape {soft_mean.shape}.")
        else:
            print("Soft-proba averaging skipped "
                  f"(available={soft_proba_available}, collected={len(all_soft_proba)}/{num_repeats}).")
    elif y_type == 'continuous':
        avg = regression_average(all_samples_arr)
        np.save(os.path.join(sampling_path, 'y_test_average.npy'), avg)

    end_time = time.time()
    print('\nTime:', end_time - start_time)

    sample_hypars = {
        "data_path": data_path,
        "diffusion_path": diffusion_path,
        "sampling_path": sampling_path,
        "batch_size": batch_size,
        "steps": steps,
        "num_repeats": num_repeats,
        "use_val": use_val,
        "base_seed": base_seed,
        "model_filename": model_filename,
        "text_pooling": text_pooling,
        "normalize_tab_cond": normalize_tab_cond,
        "normalize_text_tokens": normalize_text_tokens,
        "normalize_text_pooled": normalize_text_pooled,
        "normalize_projected_cond": normalize_projected_cond,
        "alignment_mix_alpha": alignment_mix_alpha,
        "raw_vae_dat_path": raw_vae_dat_path,
        "raw_text_path": raw_text_path,
        "time_used": end_time - start_time,
    }
    with open(os.path.join(sampling_path, 'sample_note.txt'), 'w') as file:
        metric_dict_write(file, sample_hypars, None)
