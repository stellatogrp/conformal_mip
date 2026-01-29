import argparse
import os

import gurobipy as gp
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm


def get_full_problem_data(
    lp_dir: str,
    c_changing: bool = True,
    b_changing: bool = True,
    A_changing: bool = True,
    ints_changing: bool = True,
):
    """
    Extract full problem data from an LP file.

    Args:
        lp_dir (str): Path to the LP file.
        c_changing (bool): Whether the cost vector can change.
        b_changing (bool): Whether the RHS vector can change.
        A_changing (bool): Whether the constraint matrix can change.
        ints_changing (bool): Whether the integer variable set can change.
    Returns:
        representation: np.ndarray
    """
    env = gp.Env(params={'OutputFlag': 0})
    model = gp.read(lp_dir, env=env)

    c = np.array(model.getAttr('Obj'))
    b = np.array(model.getAttr('RHS'))
    A = model.getA().toarray()
    ints = np.array(
        [1 if v.VType in {gp.GRB.INTEGER, gp.GRB.BINARY} else 0 for v in model.getVars()]
    )

    representation = {
        'n_vars': np.array([model.NumVars]),
        'n_cons': np.array([model.NumConstrs]),
        'n_ints': np.array([np.sum(ints)]),
        'n_binaries': np.array([np.sum(ints == 1)]),
        'c': c if c_changing else None,
        'b': b if b_changing else None,
        'A': A if A_changing else None,
        'ints': ints if ints_changing else None,
    }
    return representation


def get_problem_representation_dataset(
    lp_folder: str,
    output_file: str,
    c_changing: bool = True,
    b_changing: bool = True,
    A_changing: bool = True,
    ints_changing: bool = True,
    use_keys: list = ['n_vars', 'n_cons', 'n_ints', 'n_binaries', 'c', 'b', 'A', 'ints'],
):
    """
    Extract problem representation dataset from a list of LP files.

    Args:
        lp_dirs (list): List of paths to LP files.
        c_changing (bool): Whether the cost vector can change.
        b_changing (bool): Whether the RHS vector can change.
        A_changing (bool): Whether the constraint matrix can change.
        ints_changing (bool): Whether the integer variable set can change.
    Returns:
        dataset: np.ndarray
    """

    dataset = []
    names = []
    for name in tqdm(os.listdir(lp_folder)):
        if not name.endswith('.lp') and not name.endswith('.mps'):
            continue
        name_theta = name.replace('.lp', '_theta.npy').replace('.mps', '_theta.npy')
        if os.path.exists(os.path.join(lp_folder, name_theta)):
            theta = np.load(os.path.join(lp_folder, name_theta))
            dataset.append(theta)
            names.append(name)
            continue
        lp_dir = os.path.join(lp_folder, name)
        representation = get_full_problem_data(
            lp_dir,
            c_changing,
            b_changing,
            A_changing,
            ints_changing,
        )

        representation_vec = np.concatenate(
            [
                representation[key].flatten()
                for key in representation.keys()
                if representation[key] is not None and key in use_keys
            ]
        )
        np.save(os.path.join(lp_folder, name_theta), representation_vec)
        dataset.append(representation_vec.flatten())
        names.append(name)
    dataset = pd.DataFrame(dataset)
    dataset['problem_name'] = names
    dataset.set_index('problem_name', inplace=True)

    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    dataset.to_csv(output_file)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_dir",
        type=str,
        required=True,
        help="Path to config.",
    )
    parser.add_argument(
        "--full_problem_rep",
        type=bool,
        required=False,
        help="Whether to use full problem representation.",
        default=False,
    )
    parser.add_argument(
        "--c_changing",
        type=bool,
        required=False,
        help="Whether the cost vector can change.",
        default=True,
    )
    parser.add_argument(
        "--b_changing",
        type=bool,
        required=False,
        help="Whether the rhs vector can change.",
        default=True,
    )
    parser.add_argument(
        "--A_changing",
        type=bool,
        required=False,
        help="Whether the constraint matrix can change.",
        default=True,
    )
    parser.add_argument(
        "--ints_changing",
        type=bool,
        required=False,
        help="Whether the integer variable set can change.",
        default=True,
    )
    args = parser.parse_args()

    if args.full_problem_rep:
        use_keys = ['n_vars', 'n_cons', 'n_ints', 'n_binaries', 'c', 'b', 'A', 'ints']
    else:
        use_keys = ['n_vars', 'n_cons', 'n_ints', 'n_binaries']

    # load config
    with open(args.config_dir, 'r') as f:
        config = yaml.safe_load(f)

    lp_folder = config['problems']['train_dir']
    output_file = os.path.join(lp_folder, 'problem_representations.csv')
    get_problem_representation_dataset(
        lp_folder,
        output_file,
        args.c_changing,
        args.b_changing,
        args.A_changing,
        args.ints_changing,
        use_keys,
    )

    lp_folder = config['problems']['eval_dir']
    output_file = os.path.join(lp_folder, 'problem_representations.csv')
    get_problem_representation_dataset(
        lp_folder,
        output_file,
        args.c_changing,
        args.b_changing,
        args.A_changing,
        args.ints_changing,
        use_keys,
    )

    lp_folder = config['problems']['test_dir']
    output_file = os.path.join(lp_folder, 'problem_representations.csv')
    get_problem_representation_dataset(
        lp_folder,
        output_file,
        args.c_changing,
        args.b_changing,
        args.A_changing,
        args.ints_changing,
        use_keys,
    )