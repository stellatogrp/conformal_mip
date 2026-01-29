import os

import numpy as np
import pandas as pd
import torch
from cmippy.utils import (
    normalize_input,
    normalize_target,
    relative_gap,
    unnormalize_output,
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class RaggedHeightMatrixDataset(Dataset):
    def __init__(self, X_list, y=None, dtype=torch.float32):
        """
        X_list: list of array-like/tensors, each of shape (H_i, W)
        y: optional labels, length N
        """
        self.X = [torch.as_tensor(x, dtype=dtype) for x in X_list]
        W0 = self.X[0].shape[1]
        for i, x in enumerate(self.X):
            assert x.ndim == 2, f"X[{i}] must be 2D, got {x.shape}"
            assert x.shape[1] == W0, f"All W must match. X[{i}].shape={x.shape}, expected W={W0}"
        self.y = None if y is None else y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is None:
            return self.X[idx]
        return self.X[idx], self.y[idx]


def pad_height_collate(batch, pad_value=0.0, add_channel_dim=False, return_mask=True):
    """
    Pads matrices along height to max H in the batch.

    Returns:
      X_padded: (B, Hmax, W) or (B, 1, Hmax, W) if add_channel_dim
      heights:  (B,) original heights
      mask:     True where valid:
                - if add_channel_dim: (B, 1, Hmax, 1)
                - else:              (B, Hmax, 1)
      y:        batched labels if present
    """
    has_y = isinstance(batch[0], (tuple, list)) and len(batch[0]) == 2
    if has_y:
        xs, ys = zip(*batch)
        ys = ys #torch.stack([torch.as_tensor(y) for y in ys])
    else:
        xs = batch
        ys = None

    B = len(xs)
    W = xs[0].shape[1]
    heights = torch.tensor([x.shape[0] for x in xs], dtype=torch.long)
    Hmax = int(heights.max().item())

    X_padded = xs[0].new_full((B, Hmax, W), fill_value=pad_value)
    y_padded = xs[0].new_full((B, Hmax, 1), fill_value=pad_value)
    for i, (x, y) in enumerate(zip(xs, ys)):
        h = x.shape[0]
        X_padded[i, :h, :] = x
        X_padded[i, h:, :] = x[h-1]
        y_padded[i, :h, 0] = y
        y_padded[i, h:, 0] = y[-1]

    if add_channel_dim:
        X_padded = X_padded.unsqueeze(1)  # (B,1,Hmax,W)

    out = [X_padded, heights]

    if return_mask:
        mask_h = torch.arange(Hmax).unsqueeze(0) < heights.unsqueeze(1)  # (B, Hmax)
        if add_channel_dim:
            mask = mask_h[:, None, :, None]   # (B,1,Hmax,1)
        else:
            mask = mask_h[:, :, None]         # (B,Hmax,1)
        out.append(mask)

    if ys is not None:
        out.append(y_padded)

    return tuple(out)


def train(
    data_dir: str,
    model: torch.nn.Module,
    epochs: int,
    batch_size: int=1,
    device='cpu',
    loss_fn: callable=None,
    callback: callable=None,
    lr: float=1e-4,
    drop_cols = [],
    n: int=None,
    problems_dir: str = None,
    gap_type: str = 'absolute',
):
    if loss_fn is None:
        loss_fn = torch.nn.L1Loss()

    # make dataframe
    dfs = []
    ys = []
    i = 0
    files = os.listdir(data_dir)
    files = sorted(files)
    for file in tqdm(files, desc="Loading data..."):
        try:
            if (file.startswith('representations')) or (not file.endswith('.csv')):
                continue

            csv_path = os.path.join(data_dir, file)
            df = pd.read_csv(csv_path)
            y = df['true_gap']

            if gap_type == 'relative':
                y = relative_gap(df['true_gap'], df['ov'])
                if np.max(y) > 1e10:
                    print(f"Warning: max relative gap is very large: {file, np.max(y)}")
                    continue

            df = df.drop(columns=['true_gap', 'log_true_gap', 'ov', 'rel_gap'] if 'rel_gap' in df.columns else ['true_gap', 'log_true_gap', 'ov'])

            assert df.columns[0] == 'lb'

            cols_to_drop = []
            for d in drop_cols:
                cols_to_drop += [c for c in df.columns if c.startswith(d)]
            df = df.drop(columns=cols_to_drop)
            assert df.columns[1] == 'ub'

            df = torch.from_numpy(df.to_numpy())
            y = torch.from_numpy(y.to_numpy())
            dfs.append(df)
            ys.append(y)

            i += 1
            if n is not None:
                if i >= n:
                    break
        except Exception as e:
            print(f"Could not process {file}: {e}")
            assert False
            continue

    # rescale
    big_df = torch.cat(dfs, dim=0)
    big_df_sd = big_df.std(dim=0, keepdim=True) * 10
    big_df_mean = big_df.mean(dim=0, keepdim=True)
    big_df_bound_mean = big_df_mean[0, 0]
    big_df_bound_sd = big_df_sd[0, 0]
    big_df_mean[0, 1] = big_df_bound_mean
    big_df_sd[0, 1] = big_df_bound_sd
    big_ys_sd = big_df_bound_sd if gap_type == 'absolute' else torch.tensor([1.])
    big_ys_mean = torch.tensor([0.]) #big_df_bound_mean
    for i in range(len(dfs)):
        dfs[i] = normalize_input(dfs[i], big_df_mean, big_df_sd)
        dfs[i] = dfs[i].nan_to_num(0.0)
    for i in range(len(ys)):
        ys[i] = normalize_target(ys[i], big_ys_mean, big_ys_sd)
        ys[i] = ys[i].nan_to_num(0.0)

    rescaling_data = {
        'input_mean': big_df_mean,
        'input_sd': big_df_sd,
        'output_mean': big_ys_mean,
        'output_sd': big_ys_sd,
    }

    if problems_dir is None:
        return data_train_loop(
            dfs,
            ys,
            model,
            epochs,
            batch_size,
            device,
            loss_fn,
            callback,
            lr,
            rescaling_data
        )
    else:
        assert False


def data_train_loop(
        dfs,
        ys,
        model,
        epochs,
        batch_size,
        device,
        loss_fn,
        callback,
        lr,
        rescaling_data
):
    big_df_mean = rescaling_data['input_mean']
    big_df_sd = rescaling_data['input_sd']
    big_ys_mean = rescaling_data['output_mean']
    big_ys_sd = rescaling_data['output_sd']

    dataset = RaggedHeightMatrixDataset(dfs, ys)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: pad_height_collate(b, pad_value=0.0, add_channel_dim=False, return_mask=True),
    )

    params = list(model.parameters())
    if lr != 0:
        optimizer = torch.optim.Adam(params, lr=lr)

    for epoch in range(epochs):
        try: 
            # get the next batch of data
            x, height, mask, y = next(iter(loader))
            x = x.permute(1, 0, 2).to(device)
            lbs = x[:, :, 0].to(device).unsqueeze(2)
            ubs = x[:, :, 1].to(device).unsqueeze(2)

            # reset the model's hidden states (if any) (this is important for RNNs/LSTMs to avoid carrying over state between batches)
            model.reset()

            # get the model's output
            output = model(x, lbs, ubs)
            assert not torch.isnan(output).any()
            output_ts = output.permute(1, 0, 2)
            if output_ts.isnan().any():
                print("NaN values in output")
                assert False
            if output_ts.isinf().any():
                print("Inf values in output")
                assert False
            target = y.to(device)
            assert output_ts.shape == target.shape

            # unnormalize
            output_ts = unnormalize_output(output_ts, big_ys_mean.to(device), big_ys_sd.to(device))
            target = unnormalize_output(target, big_ys_mean.to(device), big_ys_sd.to(device))
            if target.isnan().any():
                print("NaN values in target after unnormalization")
                assert False
            if target.isinf().any():
                print("Inf values in target after unnormalization")
                assert False
            loss = loss_fn(output_ts, target)
            if torch.isnan(loss).any():
                print("NaN values in loss")
                assert False
            if torch.isinf(loss).any():
                print("Inf values in loss")
                assert False

            # take gradient steps
            if lr != 0:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # make sure parameters are not NaN and clamp gradients
                for param in params:
                    assert not torch.isnan(param).any()
                    if param.grad is not None:
                        param.grad.data.clamp_(-1.0, 1.0)

            # run callback (see the test script for details)
            if callback is not None:
                callback(epoch, loss.item(), model, rescaling_data, dfs[0] * (1 + big_df_sd) + big_df_mean, ys[0] * (1 + big_ys_sd) + big_ys_mean)

        except Exception as e:
            print(f"Exception during training at epoch {epoch}: {e}")
            assert False
    return model, loss.item()
