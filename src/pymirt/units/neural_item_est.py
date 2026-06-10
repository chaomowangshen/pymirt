import copy

import numpy as np

from .sparse_data import analyze_sparse_response, dataframe_to_sparse_response


NN_DEFAULT_CONFIG = {
    "epochs": 1000,
    "lr": 1e-3,
    "batch_size": None,
    "early_stopping_patience": 30,
    "early_stopping_delta": 1e-3,
    "depth": 1,
    "activation": "sigmoid",
    "embedding_dim": 128,
    "hidden_dim": 100,
    "pooling": "mean",
    "use_residual": True,
    "use_layernorm": True,
    "use_b_mlp": True,
    "use_d_mlp": True,
    "use_residual_shrinkage": True,
    "residual_init": 0.1,
    "a_upper": 3.0,
    "d_range": 3.5,
    "device": "cpu",
    "random_state": None,
    "retain_backend_model": False,
}


def neural_irt_est(response_df, model="2pl", nn_config=None, verbose=False):
    """Estimate single-dimensional 1PL/2PL parameters with a CEN-QB backend."""

    torch, nn, optim = _require_torch()
    config = _merge_nn_config(nn_config)
    _validate_nn_config(config, torch)
    if config["random_state"] is not None:
        np.random.seed(int(config["random_state"]))
        torch.manual_seed(int(config["random_state"]))

    model_name = _normalize_binary_model(model)
    sparse_response = dataframe_to_sparse_response(response_df)
    analyze_sparse_response(sparse_response, binary=True)
    _validate_no_empty_users(sparse_response)

    device = torch.device(config["device"])
    person_idx = torch.tensor(sparse_response.user_idx, dtype=torch.long, device=device)
    item_idx = torch.tensor(sparse_response.item_idx, dtype=torch.long, device=device)
    response = torch.tensor(sparse_response.values, dtype=torch.float32, device=device)
    n_users = sparse_response.n_users
    n_items = sparse_response.n_items

    net = _CENQB(
        torch=torch,
        nn=nn,
        n_items=n_items,
        n_users=n_users,
        depth=config["depth"],
        activation=config["activation"],
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        pooling=config["pooling"],
        use_residual=config["use_residual"],
        use_layernorm=config["use_layernorm"],
        use_b_mlp=config["use_b_mlp"],
        use_residual_shrinkage=config["use_residual_shrinkage"],
        residual_init=config["residual_init"],
        irt_model=model_name,
    ).to(device)
    net.set_observed_data(person_idx, item_idx, response)
    optimizer = optim.Adam(net.parameters(), lr=config["lr"])
    loss_fn = nn.BCEWithLogitsLoss()
    early_stopping = _InMemoryEarlyStopping(
        patience=config["early_stopping_patience"],
        delta=config["early_stopping_delta"],
    )
    n_obs = int(response.numel())
    batch_size = n_obs if config["batch_size"] is None else int(config["batch_size"])

    for epoch in range(int(config["epochs"])):
        loss = _train_one_epoch(
            torch, net, person_idx, item_idx, response, optimizer, loss_fn, batch_size
        )
        if not np.isfinite(loss):
            raise FloatingPointError("Neural IRT training produced a non-finite loss.")
        if verbose and (epoch == 0 or (epoch + 1) % 100 == 0):
            print(f"Neural IRT epoch {epoch + 1}/{config['epochs']}, loss={loss:.6f}")
        early_stopping(loss, net)
        if early_stopping.early_stop:
            if verbose:
                print(f"Neural IRT early stopping at epoch {epoch + 1}.")
            break

    early_stopping.restore_best(net)
    net.eval()
    with torch.no_grad():
        theta, a, b = net.compute_parameters()

    theta_est = theta.detach().cpu().numpy().reshape(-1)
    a_est = a.detach().cpu().numpy().reshape(-1)
    b_est = b.detach().cpu().numpy().reshape(-1)
    if model_name == "1pl":
        a_est = np.ones(n_items, dtype=float)

    backend_model = net if bool(config["retain_backend_model"]) else None
    return a_est.astype(float), b_est.astype(float), theta_est.astype(float), backend_model


def neural_mirt_est(response_df, Q, nn_config=None, verbose=False):
    """Estimate multidimensional binary M2PL parameters with a CEN-QB backend."""

    torch, nn, optim = _require_torch()
    config = _merge_nn_config(nn_config)
    _validate_nn_config(config, torch)
    if config["random_state"] is not None:
        np.random.seed(int(config["random_state"]))
        torch.manual_seed(int(config["random_state"]))

    sparse_response = dataframe_to_sparse_response(response_df)
    analyze_sparse_response(sparse_response, binary=True)
    _validate_no_empty_users(sparse_response)
    q_matrix = _validate_mirt_q_matrix(Q, sparse_response.n_items)

    device = torch.device(config["device"])
    person_idx = torch.tensor(sparse_response.user_idx, dtype=torch.long, device=device)
    item_idx = torch.tensor(sparse_response.item_idx, dtype=torch.long, device=device)
    response = torch.tensor(sparse_response.values, dtype=torch.float32, device=device)
    q_tensor = torch.tensor(q_matrix, dtype=torch.float32, device=device)
    n_users = sparse_response.n_users
    n_items, dim = q_matrix.shape

    net = _CENQBMulti2PL(
        torch=torch,
        nn=nn,
        n_items=n_items,
        n_users=n_users,
        dim=dim,
        q_matrix=q_tensor,
        depth=config["depth"],
        activation=config["activation"],
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        pooling=config["pooling"],
        use_residual=config["use_residual"],
        use_layernorm=config["use_layernorm"],
        use_d_mlp=config["use_d_mlp"],
        use_residual_shrinkage=config["use_residual_shrinkage"],
        residual_init=config["residual_init"],
        a_upper=config["a_upper"],
        d_range=config["d_range"],
    ).to(device)
    net.set_observed_data(person_idx, item_idx, response)
    optimizer = optim.Adam(net.parameters(), lr=config["lr"])
    loss_fn = nn.BCEWithLogitsLoss()
    early_stopping = _InMemoryEarlyStopping(
        patience=config["early_stopping_patience"],
        delta=config["early_stopping_delta"],
    )
    n_obs = int(response.numel())
    batch_size = n_obs if config["batch_size"] is None else int(config["batch_size"])

    for epoch in range(int(config["epochs"])):
        loss = _train_one_epoch(
            torch, net, person_idx, item_idx, response, optimizer, loss_fn, batch_size
        )
        if not np.isfinite(loss):
            raise FloatingPointError("Neural MIRT training produced a non-finite loss.")
        if verbose and (epoch == 0 or (epoch + 1) % 100 == 0):
            print(f"Neural MIRT epoch {epoch + 1}/{config['epochs']}, loss={loss:.6f}")
        early_stopping(loss, net)
        if early_stopping.early_stop:
            if verbose:
                print(f"Neural MIRT early stopping at epoch {epoch + 1}.")
            break

    early_stopping.restore_best(net)
    net.eval()
    with torch.no_grad():
        theta, a, d_net = net.compute_parameters()

    theta_est = theta.detach().cpu().numpy().astype(float)
    a_est = a.detach().cpu().numpy().astype(float)
    # CEN-QB uses theta @ a - d internally; PyMIRT exposes theta @ a + d.
    d_est = -d_net.detach().cpu().numpy().reshape(-1).astype(float)

    backend_model = net if bool(config["retain_backend_model"]) else None
    return a_est, d_est, theta_est, backend_model


def neural_grm_est(response_df, n_categories, grm_type="step", nn_config=None, verbose=False):
    """Estimate single-dimensional GRM parameters with a CEN-QB backend."""

    torch, nn, optim = _require_torch()
    config = _merge_nn_config(nn_config)
    _validate_nn_config(config, torch)
    if config["random_state"] is not None:
        np.random.seed(int(config["random_state"]))
        torch.manual_seed(int(config["random_state"]))

    grm_type = str(grm_type).lower()
    if grm_type not in {"step", "stand"}:
        raise ValueError("grm_type must be 'step' or 'stand'.")

    sparse_response = dataframe_to_sparse_response(response_df)
    n_categories = _validate_grm_categories(n_categories, sparse_response.n_items)
    analyze_sparse_response(sparse_response, n_categories=n_categories)
    _validate_no_empty_users(sparse_response)

    device = torch.device(config["device"])
    if grm_type == "step":
        observed = _make_ordinal_split_observed(sparse_response, n_categories)
        person_idx, pseudo_item_idx, pseudo_response = [
            torch.tensor(value, dtype=dtype, device=device)
            for value, dtype in (
                (observed["person_idx"], torch.long),
                (observed["pseudo_item_idx"], torch.long),
                (observed["response"], torch.float32),
            )
        ]
        net = _CENQBOrdinalSplit(
            torch=torch,
            nn=nn,
            n_items=sparse_response.n_items,
            n_users=sparse_response.n_users,
            pseudo_to_item=torch.tensor(observed["pseudo_to_item"], dtype=torch.long),
            pseudo_step=torch.tensor(observed["pseudo_step"], dtype=torch.long),
            item_offsets=torch.tensor(observed["item_offsets"], dtype=torch.long),
            item_max_scores=torch.tensor(n_categories - 1, dtype=torch.long),
            depth=config["depth"],
            activation=config["activation"],
            embedding_dim=config["embedding_dim"],
            hidden_dim=config["hidden_dim"],
            pooling=config["pooling"],
            use_residual=config["use_residual"],
            use_layernorm=config["use_layernorm"],
            use_b_mlp=config["use_b_mlp"],
            use_residual_shrinkage=config["use_residual_shrinkage"],
            residual_init=config["residual_init"],
            irt_model="2pl",
        ).to(device)
        net.set_observed_data(person_idx, pseudo_item_idx, pseudo_response)
        optimizer = optim.Adam(net.parameters(), lr=config["lr"])
        loss_fn = nn.BCEWithLogitsLoss()
        batch_size = (
            int(pseudo_response.numel())
            if config["batch_size"] is None
            else int(config["batch_size"])
        )
        train_epoch = lambda: _train_one_epoch(
            torch, net, person_idx, pseudo_item_idx, pseudo_response,
            optimizer, loss_fn, batch_size
        )
    else:
        person_idx = torch.tensor(sparse_response.user_idx, dtype=torch.long, device=device)
        item_idx = torch.tensor(sparse_response.item_idx, dtype=torch.long, device=device)
        score = torch.tensor(sparse_response.values.astype(np.int64), dtype=torch.long, device=device)
        net = _CENQBGRM(
            torch=torch,
            nn=nn,
            n_items=sparse_response.n_items,
            n_users=sparse_response.n_users,
            item_max_scores=torch.tensor(n_categories - 1, dtype=torch.long),
            depth=config["depth"],
            activation=config["activation"],
            embedding_dim=config["embedding_dim"],
            hidden_dim=config["hidden_dim"],
            pooling=config["pooling"],
            use_layernorm=config["use_layernorm"],
            irt_model="2pl",
        ).to(device)
        net.set_observed_data(person_idx, item_idx, score)
        optimizer = optim.Adam(net.parameters(), lr=config["lr"])
        batch_size = int(score.numel()) if config["batch_size"] is None else int(config["batch_size"])
        train_epoch = lambda: _train_one_epoch_grm(
            torch, net, person_idx, item_idx, score, optimizer, batch_size
        )

    early_stopping = _InMemoryEarlyStopping(
        patience=config["early_stopping_patience"],
        delta=config["early_stopping_delta"],
    )
    for epoch in range(int(config["epochs"])):
        loss = train_epoch()
        if not np.isfinite(loss):
            raise FloatingPointError("Neural GRM training produced a non-finite loss.")
        if verbose and (epoch == 0 or (epoch + 1) % 100 == 0):
            print(f"Neural GRM epoch {epoch + 1}/{config['epochs']}, loss={loss:.6f}")
        early_stopping(loss, net)
        if early_stopping.early_stop:
            if verbose:
                print(f"Neural GRM early stopping at epoch {epoch + 1}.")
            break

    early_stopping.restore_best(net)
    net.eval()
    with torch.no_grad():
        theta, a, b = net.compute_parameters()

    theta_est = theta.detach().cpu().numpy().reshape(-1).astype(float)
    a_est = a.detach().cpu().numpy().reshape(-1).astype(float)
    b_array = b.detach().cpu().numpy().astype(float)
    if grm_type == "step":
        b_est = _restore_ordinal_split_b_list(b_array, n_categories)
    else:
        b_est = _restore_grm_b_list(b_array, n_categories)
    backend_model = net if bool(config["retain_backend_model"]) else None
    return a_est, b_est, theta_est, backend_model


def neural_mgrm_est(
    response_df, Q, n_categories, grm_type="step", nn_config=None, verbose=False
):
    """Estimate multidimensional MGRM parameters with a CEN-QB backend."""

    torch, nn, optim = _require_torch()
    config = _merge_nn_config(nn_config)
    _validate_nn_config(config, torch)
    if config["random_state"] is not None:
        np.random.seed(int(config["random_state"]))
        torch.manual_seed(int(config["random_state"]))

    grm_type = str(grm_type).lower()
    if grm_type not in {"step", "stand"}:
        raise ValueError("grm_type must be 'step' or 'stand'.")

    sparse_response = dataframe_to_sparse_response(response_df)
    n_categories = _validate_grm_categories(n_categories, sparse_response.n_items)
    analyze_sparse_response(sparse_response, n_categories=n_categories)
    _validate_no_empty_users(sparse_response)
    q_matrix = _validate_mirt_q_matrix(Q, sparse_response.n_items)

    device = torch.device(config["device"])
    q_tensor = torch.tensor(q_matrix, dtype=torch.float32, device=device)
    n_users = sparse_response.n_users
    n_items, dim = q_matrix.shape

    if grm_type == "step":
        observed = _make_ordinal_split_observed(sparse_response, n_categories)
        person_idx, pseudo_item_idx, pseudo_response = [
            torch.tensor(value, dtype=dtype, device=device)
            for value, dtype in (
                (observed["person_idx"], torch.long),
                (observed["pseudo_item_idx"], torch.long),
                (observed["response"], torch.float32),
            )
        ]
        net = _CENQBMultiOrdinalSplit(
            torch=torch,
            nn=nn,
            n_items=n_items,
            n_users=n_users,
            dim=dim,
            q_matrix=q_tensor,
            pseudo_to_item=torch.tensor(observed["pseudo_to_item"], dtype=torch.long),
            pseudo_step=torch.tensor(observed["pseudo_step"], dtype=torch.long),
            item_offsets=torch.tensor(observed["item_offsets"], dtype=torch.long),
            item_max_scores=torch.tensor(n_categories - 1, dtype=torch.long),
            depth=config["depth"],
            activation=config["activation"],
            embedding_dim=config["embedding_dim"],
            hidden_dim=config["hidden_dim"],
            pooling=config["pooling"],
            use_residual=config["use_residual"],
            use_layernorm=config["use_layernorm"],
            use_d_mlp=config["use_d_mlp"],
            use_residual_shrinkage=config["use_residual_shrinkage"],
            residual_init=config["residual_init"],
            a_upper=config["a_upper"],
            d_range=config["d_range"],
        ).to(device)
        net.set_observed_data(person_idx, pseudo_item_idx, pseudo_response)
        optimizer = optim.Adam(net.parameters(), lr=config["lr"])
        loss_fn = nn.BCEWithLogitsLoss()
        batch_size = (
            int(pseudo_response.numel())
            if config["batch_size"] is None
            else int(config["batch_size"])
        )
        train_epoch = lambda: _train_one_epoch(
            torch,
            net,
            person_idx,
            pseudo_item_idx,
            pseudo_response,
            optimizer,
            loss_fn,
            batch_size,
        )
    else:
        person_idx = torch.tensor(sparse_response.user_idx, dtype=torch.long, device=device)
        item_idx = torch.tensor(sparse_response.item_idx, dtype=torch.long, device=device)
        score = torch.tensor(
            sparse_response.values.astype(np.int64), dtype=torch.long, device=device
        )
        net = _CENQBMultiGRM(
            torch=torch,
            nn=nn,
            n_items=n_items,
            n_users=n_users,
            dim=dim,
            q_matrix=q_tensor,
            item_max_scores=torch.tensor(n_categories - 1, dtype=torch.long),
            depth=config["depth"],
            activation=config["activation"],
            embedding_dim=config["embedding_dim"],
            hidden_dim=config["hidden_dim"],
            pooling=config["pooling"],
            use_layernorm=config["use_layernorm"],
            a_upper=config["a_upper"],
            d_range=config["d_range"],
        ).to(device)
        net.set_observed_data(person_idx, item_idx, score)
        optimizer = optim.Adam(net.parameters(), lr=config["lr"])
        batch_size = int(score.numel()) if config["batch_size"] is None else int(config["batch_size"])
        train_epoch = lambda: _train_one_epoch_grm(
            torch, net, person_idx, item_idx, score, optimizer, batch_size
        )

    early_stopping = _InMemoryEarlyStopping(
        patience=config["early_stopping_patience"],
        delta=config["early_stopping_delta"],
    )
    for epoch in range(int(config["epochs"])):
        loss = train_epoch()
        if not np.isfinite(loss):
            raise FloatingPointError("Neural MGRM training produced a non-finite loss.")
        if verbose and (epoch == 0 or (epoch + 1) % 100 == 0):
            print(f"Neural MGRM epoch {epoch + 1}/{config['epochs']}, loss={loss:.6f}")
        early_stopping(loss, net)
        if early_stopping.early_stop:
            if verbose:
                print(f"Neural MGRM early stopping at epoch {epoch + 1}.")
            break

    early_stopping.restore_best(net)
    net.eval()
    with torch.no_grad():
        theta, a, d_net = net.compute_parameters()

    theta_est = theta.detach().cpu().numpy().astype(float)
    a_est = a.detach().cpu().numpy().astype(float)
    # CEN-QB uses eta - d internally; PyMIRT exposes the MGRM intercept eta + d.
    d_array = -d_net.detach().cpu().numpy().astype(float)
    if grm_type == "step":
        d_est = _restore_ordinal_split_b_list(d_array, n_categories)
    else:
        d_est = _restore_grm_b_list(d_array, n_categories)

    backend_model = net if bool(config["retain_backend_model"]) else None
    return a_est, d_est, theta_est, backend_model


def _require_torch():
    try:
        import torch
        from torch import nn, optim
    except ImportError as exc:
        raise ImportError(
            "Neural estimation requires PyTorch. Install with: pip install pymirt[nn]"
        ) from exc
    return torch, nn, optim


def _merge_nn_config(nn_config):
    if nn_config is None:
        nn_config = {}
    if not isinstance(nn_config, dict):
        raise ValueError("nn_config must be a dictionary or None.")
    unknown = set(nn_config) - set(NN_DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f"Unknown nn_config keys: {sorted(unknown)}")
    config = NN_DEFAULT_CONFIG.copy()
    config.update(nn_config)
    return config


def _validate_nn_config(config, torch):
    positive_ints = ["epochs", "early_stopping_patience", "depth", "embedding_dim", "hidden_dim"]
    for key in positive_ints:
        if int(config[key]) < 1:
            raise ValueError(f"nn_config['{key}'] must be a positive integer.")
        config[key] = int(config[key])
    if config["batch_size"] is not None and int(config["batch_size"]) < 1:
        raise ValueError("nn_config['batch_size'] must be None or a positive integer.")
    if float(config["lr"]) <= 0:
        raise ValueError("nn_config['lr'] must be positive.")
    if float(config["early_stopping_delta"]) < 0:
        raise ValueError("nn_config['early_stopping_delta'] must be non-negative.")
    if float(config["a_upper"]) <= 0:
        raise ValueError("nn_config['a_upper'] must be positive.")
    if float(config["d_range"]) <= 0:
        raise ValueError("nn_config['d_range'] must be positive.")
    if config["activation"] not in {"sigmoid", "relu", "leakyrelu", "tanh"}:
        raise ValueError("nn_config['activation'] must be sigmoid, relu, leakyrelu, or tanh.")
    if config["pooling"] not in {"mean", "sum", "mean_pre", "mean_sum"}:
        raise ValueError("nn_config['pooling'] must be mean, sum, mean_pre, or mean_sum.")
    if str(config["device"]).startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA was requested in nn_config['device'], but it is not available.")
    config["lr"] = float(config["lr"])
    config["early_stopping_delta"] = float(config["early_stopping_delta"])
    config["residual_init"] = float(config["residual_init"])
    config["a_upper"] = float(config["a_upper"])
    config["d_range"] = float(config["d_range"])


def _normalize_binary_model(model):
    model = str(model).lower()
    if model == "2pl":
        return "2pl"
    if model in {"1pl", "rasch"}:
        return "1pl"
    raise NotImplementedError("Neural IRT currently supports only 1PL/Rasch and 2PL.")


def _validate_no_empty_users(sparse_response):
    counts = np.bincount(sparse_response.user_idx, minlength=sparse_response.n_users)
    empty = np.flatnonzero(counts == 0)
    if empty.size:
        raise ValueError(
            "Users with no observed responses are not supported: "
            + ", ".join(map(str, empty.tolist()))
        )


def _validate_mirt_q_matrix(Q, n_items):
    q_matrix = np.asarray(Q, dtype=float)
    if q_matrix.ndim != 2:
        raise ValueError("Q must be a 2-D matrix.")
    if q_matrix.shape[0] != n_items:
        raise ValueError(
            f"Q row count ({q_matrix.shape[0]}) must match the number of items ({n_items})."
        )
    if q_matrix.shape[1] < 1:
        raise ValueError("Q must contain at least one dimension.")
    if not np.all(np.isfinite(q_matrix)):
        raise ValueError("Q must contain finite numeric values.")
    if not np.all(np.isin(q_matrix, [0.0, 1.0])):
        raise ValueError("Q must be a binary 0/1 matrix for neural MIRT estimation.")
    if np.any(q_matrix.sum(axis=1) == 0):
        raise ValueError("Each Q row must contain at least one active dimension.")

    identity = np.eye(q_matrix.shape[1])
    contains_identity = all(
        np.any(np.all(q_matrix == identity_row, axis=1))
        for identity_row in identity
    )
    if not contains_identity:
        raise ValueError("Q must contain an identity matrix.")
    return q_matrix.astype(float, copy=True)


def _validate_grm_categories(n_categories, n_items):
    if n_categories is None:
        raise ValueError("n_categories is required for neural GRM estimation.")
    n_categories = np.asarray(n_categories, dtype=np.int64)
    if n_categories.ndim != 1 or n_categories.size != n_items:
        raise ValueError("n_categories must be a 1-D array with one entry per item.")
    if np.any(n_categories < 2):
        raise ValueError("Each GRM item must have at least two categories.")
    return n_categories


def _make_ordinal_split_observed(sparse_response, n_categories):
    item_max_scores = n_categories - 1
    item_offsets = np.concatenate(([0], np.cumsum(item_max_scores))).astype(np.int64)
    n_pseudo_items = int(item_offsets[-1])
    if n_pseudo_items < 1:
        raise ValueError("No ordinal pseudo-items were created.")

    pseudo_to_item = np.repeat(np.arange(sparse_response.n_items, dtype=np.int64), item_max_scores)
    pseudo_step = np.concatenate(
        [
            np.arange(1, int(max_score) + 1, dtype=np.int64)
            for max_score in item_max_scores
        ]
    )
    person_parts = []
    pseudo_parts = []
    response_parts = []
    scores = sparse_response.values.astype(np.int64)

    for item_id in range(sparse_response.n_items):
        max_score = int(item_max_scores[item_id])
        steps = np.arange(1, max_score + 1, dtype=np.int64)
        pseudo_ids = item_offsets[item_id] + steps - 1
        observed_ids = sparse_response.item_observations(item_id)
        persons = sparse_response.user_idx[observed_ids]
        item_scores = scores[observed_ids]
        person_parts.append(np.repeat(persons, max_score))
        pseudo_parts.append(np.tile(pseudo_ids, persons.size))
        response_parts.append(
            (item_scores.reshape(-1, 1) >= steps.reshape(1, -1))
            .astype(float)
            .reshape(-1)
        )

    return {
        "person_idx": np.concatenate(person_parts),
        "pseudo_item_idx": np.concatenate(pseudo_parts),
        "response": np.concatenate(response_parts),
        "pseudo_to_item": pseudo_to_item,
        "pseudo_step": pseudo_step,
        "item_offsets": item_offsets,
    }


def _restore_grm_b_list(b_array, n_categories):
    b_list = []
    for item_id, n_cat in enumerate(n_categories):
        n_thresholds = int(n_cat) - 1
        b_list.append(np.asarray(b_array[item_id, :n_thresholds], dtype=float).copy())
    return b_list


def _restore_ordinal_split_b_list(b_array, n_categories):
    flat_b = np.asarray(b_array, dtype=float).reshape(-1)
    item_max_scores = np.asarray(n_categories, dtype=np.int64) - 1
    item_offsets = np.concatenate(([0], np.cumsum(item_max_scores))).astype(np.int64)
    return [
        flat_b[item_offsets[item_id] : item_offsets[item_id + 1]].copy()
        for item_id in range(item_max_scores.size)
    ]


def _train_one_epoch(torch, model, person_idx, item_idx, response, optimizer, loss_fn, batch_size):
    model.train()
    n_obs = int(response.numel())
    perm = torch.randperm(n_obs, device=response.device)
    total_loss = 0.0
    total_count = 0
    for start in range(0, n_obs, batch_size):
        batch_ids = perm[start : start + batch_size]
        optimizer.zero_grad()
        logits = model(person_idx[batch_ids], item_idx[batch_ids])
        batch_response = response[batch_ids]
        loss = loss_fn(logits, batch_response)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * int(batch_response.numel())
        total_count += int(batch_response.numel())
    return total_loss / max(total_count, 1)


def _train_one_epoch_grm(torch, model, person_idx, item_idx, score, optimizer, batch_size):
    model.train()
    n_obs = int(score.numel())
    perm = torch.randperm(n_obs, device=score.device)
    total_loss = 0.0
    total_count = 0
    for start in range(0, n_obs, batch_size):
        batch_ids = perm[start : start + batch_size]
        optimizer.zero_grad()
        batch_score = score[batch_ids]
        loss = model.neg_log_likelihood(
            person_idx[batch_ids], item_idx[batch_ids], batch_score
        )
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * int(batch_score.numel())
        total_count += int(batch_score.numel())
    return total_loss / max(total_count, 1)


class _InMemoryEarlyStopping:
    def __init__(self, patience=30, delta=1e-3):
        self.patience = int(patience)
        self.delta = float(delta)
        self.best_score = None
        self.epochs_no_improve = 0
        self.early_stop = False
        self.best_state = None

    def __call__(self, loss, model):
        score = -float(loss)
        if self.best_score is None or score >= self.best_score + self.delta:
            self.best_score = score
            self.epochs_no_improve = 0
            self.best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            self.epochs_no_improve += 1
            if self.epochs_no_improve >= self.patience:
                self.early_stop = True

    def restore_best(self, model):
        if self.best_state is not None:
            model.load_state_dict(copy.deepcopy(self.best_state))


def _activation_layer(nn, name):
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "relu":
        return nn.ReLU()
    if name == "leakyrelu":
        return nn.LeakyReLU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")


def _make_mlp(nn, input_dim, hidden_dim, depth, activation):
    layers = nn.ModuleList()
    layers.append(nn.Linear(input_dim, hidden_dim))
    for _ in range((depth - 1) // 2):
        layers.append(nn.Linear(hidden_dim, hidden_dim * 2))
        layers.append(nn.Linear(hidden_dim * 2, hidden_dim))
    return layers, _activation_layer(nn, activation)


class _PersonNetBase:
    def _pool(self, torch, x, group_idx, n_groups, layers, activation, pooling):
        counts = None
        if pooling in ("mean_pre", "mean_sum"):
            pooled_sum = torch.zeros(n_groups, x.shape[1], dtype=x.dtype, device=x.device)
            pooled_sum.index_add_(0, group_idx, x)
            counts = torch.zeros(n_groups, 1, dtype=x.dtype, device=x.device)
            counts.index_add_(0, group_idx, torch.ones(x.shape[0], 1, dtype=x.dtype, device=x.device))
            pooled_mean = pooled_sum / counts.clamp_min(1.0)
            x = torch.cat([pooled_mean, pooled_sum], dim=1) if pooling == "mean_sum" else pooled_mean
            for layer in layers:
                x = activation(layer(x))
            pooled = x
        else:
            for layer in layers:
                x = activation(layer(x))
            pooled = torch.zeros(n_groups, x.shape[1], dtype=x.dtype, device=x.device)
            pooled.index_add_(0, group_idx, x)
        return pooled, counts


def _make_person_net(torch, nn):
    class PersonNet(nn.Module):
        def __init__(
            self, num_items, depth, activation, hidden_dim, embedding_dim, pooling,
            use_residual, use_layernorm, use_residual_shrinkage, residual_init
        ):
            super().__init__()
            self.pooling = pooling
            self.use_residual = use_residual
            self.use_layernorm = use_layernorm
            self.use_residual_shrinkage = use_residual_shrinkage
            self.item_response_embedding = nn.Embedding(num_items * 2, embedding_dim)
            self.item_response_residual = nn.Embedding(num_items * 2, 1) if use_residual else None
            self.residual_alpha = (
                nn.Parameter(torch.tensor(float(residual_init)))
                if use_residual and use_residual_shrinkage
                else None
            )
            mlp_input_dim = embedding_dim * 2 if pooling == "mean_sum" else embedding_dim
            self.layers, self.activation = _make_mlp(nn, mlp_input_dim, hidden_dim, depth, activation)
            self.norm = nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity()
            self.linear = nn.Linear(hidden_dim, 1)
            self.tanh = nn.Tanh()

        def forward(self, item_idx, response, person_idx, n_person):
            response_idx = response.long().view(-1)
            if torch.any((response_idx < 0) | (response_idx > 1)):
                raise ValueError("Neural IRT expects binary responses coded as 0/1.")
            embedding_idx = item_idx * 2 + response_idx
            x = self.item_response_embedding(embedding_idx)
            if self.use_residual:
                residual_token = self.item_response_residual(embedding_idx)

            counts = None
            if self.pooling in ("mean_pre", "mean_sum"):
                pooled_sum = torch.zeros(
                    n_person, x.shape[1], dtype=x.dtype, device=x.device
                )
                pooled_sum.index_add_(0, person_idx, x)
                counts = torch.zeros(n_person, 1, dtype=x.dtype, device=x.device)
                counts.index_add_(
                    0, person_idx,
                    torch.ones(x.shape[0], 1, dtype=x.dtype, device=x.device)
                )
                pooled_mean = pooled_sum / counts.clamp_min(1.0)
                x = (
                    torch.cat([pooled_mean, pooled_sum], dim=1)
                    if self.pooling == "mean_sum"
                    else pooled_mean
                )
                for layer in self.layers:
                    x = self.activation(layer(x))
                pooled = x
            else:
                for layer in self.layers:
                    x = self.activation(layer(x))
                pooled = torch.zeros(
                    n_person, x.shape[1], dtype=x.dtype, device=x.device
                )
                pooled.index_add_(0, person_idx, x)

            if self.use_residual:
                residual = torch.zeros(n_person, 1, dtype=x.dtype, device=x.device)
                residual.index_add_(0, person_idx, residual_token)

            if self.pooling == "mean":
                if counts is None:
                    counts = torch.zeros(n_person, 1, dtype=x.dtype, device=x.device)
                    counts.index_add_(
                        0, person_idx,
                        torch.ones(x.shape[0], 1, dtype=x.dtype, device=x.device)
                    )
                pooled = pooled / counts.clamp_min(1.0)
                if self.use_residual:
                    residual = residual / counts.clamp_min(1.0)
            elif self.pooling == "mean_sum":
                if self.use_residual:
                    residual = residual / counts.clamp_min(1.0)
            elif self.pooling == "mean_pre":
                if self.use_residual:
                    residual = residual / counts.clamp_min(1.0)
            elif self.pooling != "sum":
                raise ValueError(f"Unsupported pooling: {self.pooling}")

            pooled = self.norm(pooled)
            z = self.tanh(self.linear(pooled)) * 3
            if self.use_residual:
                z = z + (self.residual_alpha * residual if self.residual_alpha is not None else residual)
            if z.shape[0] > 1:
                z = (z - z.mean()) / (z.std(unbiased=False) + 1e-6)
            return z

    return PersonNet


def _make_item_net(torch, nn):
    class ItemNet(nn.Module):
        def __init__(
            self, num_users, depth, activation, hidden_dim, embedding_dim, pooling,
            use_residual, use_layernorm, use_b_mlp, use_residual_shrinkage, residual_init
        ):
            super().__init__()
            self.pooling = pooling
            self.use_residual = use_residual
            self.use_layernorm = use_layernorm
            self.use_b_mlp = use_b_mlp
            self.use_residual_shrinkage = use_residual_shrinkage
            self.person_response_embedding = nn.Embedding(num_users * 2, embedding_dim)
            self.person_response_residual = nn.Embedding(num_users * 2, 1) if use_residual else None
            self.residual_alpha = (
                nn.Parameter(torch.tensor(float(residual_init)))
                if use_residual and use_residual_shrinkage
                else None
            )
            mlp_input_dim = embedding_dim * 2 if pooling == "mean_sum" else embedding_dim
            self.layers, self.activation = _make_mlp(nn, mlp_input_dim, hidden_dim, depth, activation)
            self.norm = nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity()
            self.linear_a = nn.Linear(hidden_dim, 1)
            if use_b_mlp:
                b_hidden_dim = max(1, hidden_dim // 2)
                self.b_head = nn.Sequential(
                    nn.Linear(hidden_dim, b_hidden_dim),
                    _activation_layer(nn, activation),
                    nn.Linear(b_hidden_dim, 1),
                )
            else:
                self.b_head = nn.Linear(hidden_dim, 1)
            self.tanh = nn.Tanh()
            self.sig = nn.Sigmoid()

        def forward(self, person_idx, response, item_idx, n_item):
            response_idx = response.long().view(-1)
            if torch.any((response_idx < 0) | (response_idx > 1)):
                raise ValueError("Neural IRT expects binary responses coded as 0/1.")
            embedding_idx = person_idx * 2 + response_idx
            x = self.person_response_embedding(embedding_idx)
            if self.use_residual:
                residual_token = self.person_response_residual(embedding_idx)

            counts = None
            if self.pooling in ("mean_pre", "mean_sum"):
                pooled_sum = torch.zeros(
                    n_item, x.shape[1], dtype=x.dtype, device=x.device
                )
                pooled_sum.index_add_(0, item_idx, x)
                counts = torch.zeros(n_item, 1, dtype=x.dtype, device=x.device)
                counts.index_add_(
                    0, item_idx,
                    torch.ones(x.shape[0], 1, dtype=x.dtype, device=x.device)
                )
                pooled_mean = pooled_sum / counts.clamp_min(1.0)
                x = (
                    torch.cat([pooled_mean, pooled_sum], dim=1)
                    if self.pooling == "mean_sum"
                    else pooled_mean
                )
                for layer in self.layers:
                    x = self.activation(layer(x))
                pooled = x
            else:
                for layer in self.layers:
                    x = self.activation(layer(x))
                pooled = torch.zeros(
                    n_item, x.shape[1], dtype=x.dtype, device=x.device
                )
                pooled.index_add_(0, item_idx, x)

            if self.use_residual:
                residual = torch.zeros(n_item, 1, dtype=x.dtype, device=x.device)
                residual.index_add_(0, item_idx, residual_token)

            if self.pooling == "mean":
                if counts is None:
                    counts = torch.zeros(n_item, 1, dtype=x.dtype, device=x.device)
                    counts.index_add_(
                        0, item_idx,
                        torch.ones(x.shape[0], 1, dtype=x.dtype, device=x.device)
                    )
                pooled = pooled / counts.clamp_min(1.0)
                if self.use_residual:
                    residual = residual / counts.clamp_min(1.0)
            elif self.pooling == "mean_sum":
                if self.use_residual:
                    residual = residual / counts.clamp_min(1.0)
            elif self.pooling == "mean_pre":
                if self.use_residual:
                    residual = residual / counts.clamp_min(1.0)
            elif self.pooling != "sum":
                raise ValueError(f"Unsupported pooling: {self.pooling}")

            pooled = self.norm(pooled)
            raw_a = self.linear_a(pooled)
            a = self.sig(raw_a) * 3 + 1e-6
            b = self.tanh(self.b_head(pooled)) * 3
            if self.use_residual:
                b = b + (self.residual_alpha * residual if self.residual_alpha is not None else residual)
            return a, b

    return ItemNet


class _CENQB:
    def __new__(cls, torch, nn, **kwargs):
        PersonNet = _make_person_net(torch, nn)
        ItemNet = _make_item_net(torch, nn)

        class CENQB(nn.Module):
            def __init__(
                self, n_items, n_users, depth, activation, embedding_dim, hidden_dim,
                pooling, use_residual, use_layernorm, use_b_mlp,
                use_residual_shrinkage, residual_init, irt_model
            ):
                super().__init__()
                self.n_item = n_items
                self.n_person = n_users
                self.irt_model = irt_model
                self.person_net = PersonNet(
                    n_items, depth, activation, hidden_dim, embedding_dim, pooling,
                    use_residual, use_layernorm, use_residual_shrinkage, residual_init
                )
                self.item_net = ItemNet(
                    n_users, depth, activation, hidden_dim, embedding_dim, pooling,
                    use_residual, use_layernorm, use_b_mlp,
                    use_residual_shrinkage, residual_init
                )
                self.register_buffer("_observed_person_idx", torch.empty(0, dtype=torch.long), persistent=False)
                self.register_buffer("_observed_item_idx", torch.empty(0, dtype=torch.long), persistent=False)
                self.register_buffer("_observed_response", torch.empty(0, dtype=torch.float32), persistent=False)

            def set_observed_data(self, person_idx, item_idx, response):
                device = next(self.parameters()).device
                self._observed_person_idx = person_idx.long().to(device)
                self._observed_item_idx = item_idx.long().to(device)
                self._observed_response = response.float().to(device)

            def compute_parameters(self):
                if self._observed_response.numel() == 0:
                    raise RuntimeError("Observed response data has not been set.")
                z = self.person_net(
                    self._observed_item_idx,
                    self._observed_response,
                    self._observed_person_idx,
                    self.n_person,
                )
                a, b = self.item_net(
                    self._observed_person_idx,
                    self._observed_response,
                    self._observed_item_idx,
                    self.n_item,
                )
                if self.irt_model == "1pl":
                    a = torch.ones(self.n_item, 1, dtype=b.dtype, device=b.device)
                return z, a, b

            def forward(self, person_idx, item_idx):
                z, a, b = self.compute_parameters()
                logits = a[item_idx] * (z[person_idx] - b[item_idx])
                return logits.flatten()

        return CENQB(**kwargs)


def _make_person_net_multi(torch, nn):
    class PersonNetMulti(nn.Module):
        def __init__(
            self, num_items, dim, depth, activation, hidden_dim, embedding_dim,
            pooling, use_residual, use_layernorm, use_residual_shrinkage,
            residual_init
        ):
            super().__init__()
            self.dim = int(dim)
            self.pooling = pooling
            self.use_residual = use_residual
            self.use_layernorm = use_layernorm
            self.use_residual_shrinkage = use_residual_shrinkage
            self.item_response_embedding = nn.Embedding(num_items * 2, embedding_dim)
            self.item_response_residual = (
                nn.Embedding(num_items * 2, self.dim) if use_residual else None
            )
            self.residual_alpha = (
                nn.Parameter(torch.tensor(float(residual_init)))
                if use_residual and use_residual_shrinkage
                else None
            )
            mlp_input_dim = embedding_dim * 2 if pooling == "mean_sum" else embedding_dim
            self.layers, self.activation = _make_mlp(
                nn, mlp_input_dim, hidden_dim, depth, activation
            )
            self.norm = nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity()
            self.linear = nn.Linear(hidden_dim, self.dim)
            self.tanh = nn.Tanh()

        def forward(self, item_idx, response, person_idx, n_person):
            response_idx = response.long().view(-1)
            if torch.any((response_idx < 0) | (response_idx > 1)):
                raise ValueError("Neural MIRT expects binary responses coded as 0/1.")

            embedding_idx = item_idx * 2 + response_idx
            x = self.item_response_embedding(embedding_idx)
            if self.use_residual:
                residual_token = self.item_response_residual(embedding_idx)

            counts = None
            if self.pooling in ("mean_pre", "mean_sum"):
                pooled_sum = torch.zeros(
                    n_person, x.shape[1], dtype=x.dtype, device=x.device
                )
                pooled_sum.index_add_(0, person_idx, x)
                counts = torch.zeros(n_person, 1, dtype=x.dtype, device=x.device)
                counts.index_add_(
                    0, person_idx,
                    torch.ones(x.shape[0], 1, dtype=x.dtype, device=x.device)
                )
                pooled_mean = pooled_sum / counts.clamp_min(1.0)
                x = (
                    torch.cat([pooled_mean, pooled_sum], dim=1)
                    if self.pooling == "mean_sum"
                    else pooled_mean
                )
                for layer in self.layers:
                    x = self.activation(layer(x))
                pooled = x
            else:
                for layer in self.layers:
                    x = self.activation(layer(x))
                pooled = torch.zeros(
                    n_person, x.shape[1], dtype=x.dtype, device=x.device
                )
                pooled.index_add_(0, person_idx, x)

            if self.use_residual:
                residual = torch.zeros(
                    n_person, self.dim, dtype=x.dtype, device=x.device
                )
                residual.index_add_(0, person_idx, residual_token)

            if self.pooling == "mean":
                if counts is None:
                    counts = torch.zeros(n_person, 1, dtype=x.dtype, device=x.device)
                    counts.index_add_(
                        0, person_idx,
                        torch.ones(x.shape[0], 1, dtype=x.dtype, device=x.device)
                    )
                pooled = pooled / counts.clamp_min(1.0)
                if self.use_residual:
                    residual = residual / counts.clamp_min(1.0)
            elif self.pooling == "mean_sum":
                if self.use_residual:
                    residual = residual / counts.clamp_min(1.0)
            elif self.pooling == "mean_pre":
                if self.use_residual:
                    residual = residual / counts.clamp_min(1.0)
            elif self.pooling != "sum":
                raise ValueError(f"Unsupported pooling: {self.pooling}")

            pooled = self.norm(pooled)
            z = self.tanh(self.linear(pooled)) * 3
            if self.use_residual:
                z = z + (
                    self.residual_alpha * residual
                    if self.residual_alpha is not None
                    else residual
                )
            if z.shape[0] > 1:
                z = (z - z.mean(dim=0, keepdim=True)) / (
                    z.std(dim=0, unbiased=False, keepdim=True) + 1e-6
                )
            return z

    return PersonNetMulti


def _make_item_net_multi(torch, nn):
    class ItemNetMulti(nn.Module):
        def __init__(
            self, num_users, dim, depth, activation, hidden_dim, embedding_dim,
            pooling, use_residual, use_layernorm, use_d_mlp,
            use_residual_shrinkage, residual_init, a_upper, d_range
        ):
            super().__init__()
            self.dim = int(dim)
            self.pooling = pooling
            self.use_residual = use_residual
            self.use_layernorm = use_layernorm
            self.use_d_mlp = use_d_mlp
            self.use_residual_shrinkage = use_residual_shrinkage
            self.a_upper = float(a_upper)
            self.d_range = float(d_range)
            self.person_response_embedding = nn.Embedding(num_users * 2, embedding_dim)
            self.person_response_residual = (
                nn.Embedding(num_users * 2, 1) if use_residual else None
            )
            self.residual_alpha = (
                nn.Parameter(torch.tensor(float(residual_init)))
                if use_residual and use_residual_shrinkage
                else None
            )
            mlp_input_dim = embedding_dim * 2 if pooling == "mean_sum" else embedding_dim
            self.layers, self.activation = _make_mlp(
                nn, mlp_input_dim, hidden_dim, depth, activation
            )
            self.norm = nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity()
            self.linear_a = nn.Linear(hidden_dim, self.dim)
            if use_d_mlp:
                d_hidden_dim = max(1, hidden_dim // 2)
                self.d_head = nn.Sequential(
                    nn.Linear(hidden_dim, d_hidden_dim),
                    _activation_layer(nn, activation),
                    nn.Linear(d_hidden_dim, 1),
                )
            else:
                self.d_head = nn.Linear(hidden_dim, 1)
            self.tanh = nn.Tanh()
            self.sig = nn.Sigmoid()

        def forward(self, person_idx, response, item_idx, n_item):
            response_idx = response.long().view(-1)
            if torch.any((response_idx < 0) | (response_idx > 1)):
                raise ValueError("Neural MIRT expects binary responses coded as 0/1.")

            embedding_idx = person_idx * 2 + response_idx
            x = self.person_response_embedding(embedding_idx)
            if self.use_residual:
                residual_token = self.person_response_residual(embedding_idx)

            counts = None
            if self.pooling in ("mean_pre", "mean_sum"):
                pooled_sum = torch.zeros(n_item, x.shape[1], dtype=x.dtype, device=x.device)
                pooled_sum.index_add_(0, item_idx, x)
                counts = torch.zeros(n_item, 1, dtype=x.dtype, device=x.device)
                counts.index_add_(
                    0, item_idx,
                    torch.ones(x.shape[0], 1, dtype=x.dtype, device=x.device)
                )
                pooled_mean = pooled_sum / counts.clamp_min(1.0)
                x = (
                    torch.cat([pooled_mean, pooled_sum], dim=1)
                    if self.pooling == "mean_sum"
                    else pooled_mean
                )
                for layer in self.layers:
                    x = self.activation(layer(x))
                pooled = x
            else:
                for layer in self.layers:
                    x = self.activation(layer(x))
                pooled = torch.zeros(n_item, x.shape[1], dtype=x.dtype, device=x.device)
                pooled.index_add_(0, item_idx, x)

            if self.use_residual:
                residual = torch.zeros(n_item, 1, dtype=x.dtype, device=x.device)
                residual.index_add_(0, item_idx, residual_token)

            if self.pooling == "mean":
                if counts is None:
                    counts = torch.zeros(n_item, 1, dtype=x.dtype, device=x.device)
                    counts.index_add_(
                        0, item_idx,
                        torch.ones(x.shape[0], 1, dtype=x.dtype, device=x.device)
                    )
                pooled = pooled / counts.clamp_min(1.0)
                if self.use_residual:
                    residual = residual / counts.clamp_min(1.0)
            elif self.pooling == "mean_sum":
                if self.use_residual:
                    residual = residual / counts.clamp_min(1.0)
            elif self.pooling == "mean_pre":
                if self.use_residual:
                    residual = residual / counts.clamp_min(1.0)
            elif self.pooling != "sum":
                raise ValueError(f"Unsupported pooling: {self.pooling}")

            pooled = self.norm(pooled)
            a = self.sig(self.linear_a(pooled)) * self.a_upper + 1e-6
            d = self.tanh(self.d_head(pooled)) * self.d_range
            if self.use_residual:
                d = d + (
                    self.residual_alpha * residual
                    if self.residual_alpha is not None
                    else residual
                )
            return a, d

    return ItemNetMulti


class _CENQBMulti2PL:
    def __new__(cls, torch, nn, **kwargs):
        PersonNetMulti = _make_person_net_multi(torch, nn)
        ItemNetMulti = _make_item_net_multi(torch, nn)

        class CENQBMulti2PL(nn.Module):
            def __init__(
                self, n_items, n_users, dim, q_matrix, depth, activation,
                embedding_dim, hidden_dim, pooling, use_residual, use_layernorm,
                use_d_mlp, use_residual_shrinkage, residual_init, a_upper,
                d_range
            ):
                super().__init__()
                self.n_item = int(n_items)
                self.n_person = int(n_users)
                self.dim = int(dim)
                self.person_net = PersonNetMulti(
                    self.n_item, self.dim, depth, activation, hidden_dim,
                    embedding_dim, pooling, use_residual, use_layernorm,
                    use_residual_shrinkage, residual_init
                )
                self.item_net = ItemNetMulti(
                    self.n_person, self.dim, depth, activation, hidden_dim,
                    embedding_dim, pooling, use_residual, use_layernorm,
                    use_d_mlp, use_residual_shrinkage, residual_init,
                    a_upper, d_range
                )
                self.register_buffer("q_matrix", q_matrix.float())
                self.register_buffer("_observed_person_idx", torch.empty(0, dtype=torch.long), persistent=False)
                self.register_buffer("_observed_item_idx", torch.empty(0, dtype=torch.long), persistent=False)
                self.register_buffer("_observed_response", torch.empty(0, dtype=torch.float32), persistent=False)

            def set_observed_data(self, person_idx, item_idx, response):
                device = next(self.parameters()).device
                self._observed_person_idx = person_idx.long().to(device)
                self._observed_item_idx = item_idx.long().to(device)
                self._observed_response = response.float().to(device)

            def compute_parameters(self):
                if self._observed_response.numel() == 0:
                    raise RuntimeError("Observed response data has not been set.")
                z = self.person_net(
                    self._observed_item_idx,
                    self._observed_response,
                    self._observed_person_idx,
                    self.n_person,
                )
                a, d = self.item_net(
                    self._observed_person_idx,
                    self._observed_response,
                    self._observed_item_idx,
                    self.n_item,
                )
                q_matrix = self.q_matrix.to(device=d.device, dtype=d.dtype)
                a = a * q_matrix
                return z, a, d

            def forward(self, person_idx, item_idx):
                device = self._observed_response.device
                person_idx = person_idx.long().to(device)
                item_idx = item_idx.long().to(device)
                z, a, d = self.compute_parameters()
                logits = (z[person_idx] * a[item_idx]).sum(dim=1) - d[item_idx].view(-1)
                return logits

        return CENQBMulti2PL(**kwargs)


def _make_item_net_ordinal_split(torch, nn):
    class ItemNetOrdinalSplit(nn.Module):
        def __init__(
            self, num_users, depth, activation, hidden_dim, embedding_dim, pooling,
            use_residual, use_layernorm, use_b_mlp, use_residual_shrinkage, residual_init
        ):
            super().__init__()
            self.pooling = pooling
            self.use_residual = use_residual
            self.person_response_embedding = nn.Embedding(num_users * 2, embedding_dim)
            self.person_response_residual = nn.Embedding(num_users * 2, 1) if use_residual else None
            self.residual_alpha = (
                nn.Parameter(torch.tensor(float(residual_init)))
                if use_residual and use_residual_shrinkage
                else None
            )
            mlp_input_dim = embedding_dim * 2 if pooling == "mean_sum" else embedding_dim
            self.layers, self.activation = _make_mlp(nn, mlp_input_dim, hidden_dim, depth, activation)
            self.norm = nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity()
            self.linear_a = nn.Linear(hidden_dim, 1)
            if use_b_mlp:
                b_hidden_dim = max(1, hidden_dim // 2)
                self.b_head = nn.Sequential(
                    nn.Linear(hidden_dim, b_hidden_dim),
                    _activation_layer(nn, activation),
                    nn.Linear(b_hidden_dim, 1),
                )
            else:
                self.b_head = nn.Linear(hidden_dim, 1)
            self.tanh = nn.Tanh()
            self.sigmoid = nn.Sigmoid()

        def _apply_layers(self, x):
            for layer in self.layers:
                x = self.activation(layer(x))
            return x

        def _pool_raw_then_mlp(self, x, group_idx, n_group):
            pooled_sum = torch.zeros(n_group, x.shape[1], dtype=x.dtype, device=x.device)
            pooled_sum.index_add_(0, group_idx, x)
            counts = torch.zeros(n_group, 1, dtype=x.dtype, device=x.device)
            counts.index_add_(
                0, group_idx,
                torch.ones(x.shape[0], 1, dtype=x.dtype, device=x.device)
            )
            pooled_mean = pooled_sum / counts.clamp_min(1.0)
            pooled = (
                torch.cat([pooled_mean, pooled_sum], dim=1)
                if self.pooling == "mean_sum"
                else pooled_mean
            )
            return self._apply_layers(pooled), counts

        def _pool_hidden(self, x, group_idx, n_group):
            pooled = torch.zeros(n_group, x.shape[1], dtype=x.dtype, device=x.device)
            pooled.index_add_(0, group_idx, x)
            counts = torch.zeros(n_group, 1, dtype=x.dtype, device=x.device)
            counts.index_add_(
                0, group_idx,
                torch.ones(x.shape[0], 1, dtype=x.dtype, device=x.device)
            )
            if self.pooling == "mean":
                pooled = pooled / counts.clamp_min(1.0)
            elif self.pooling != "sum":
                raise ValueError(f"Unsupported pooling: {self.pooling}")
            return pooled, counts

        def forward(self, person_idx, response, pseudo_item_idx, original_item_idx, n_item, n_pseudo_item):
            response_idx = response.long().view(-1)
            if torch.any((response_idx < 0) | (response_idx > 1)):
                raise ValueError("Neural ordinal split expects binary pseudo-responses.")
            embedding_idx = person_idx * 2 + response_idx
            x = self.person_response_embedding(embedding_idx)
            residual_token = self.person_response_residual(embedding_idx) if self.use_residual else None

            if self.pooling in ("mean_pre", "mean_sum"):
                pooled_a, _ = self._pool_raw_then_mlp(x, original_item_idx, n_item)
                pooled_b, counts_b = self._pool_raw_then_mlp(
                    x, pseudo_item_idx, n_pseudo_item
                )
            else:
                x = self._apply_layers(x)
                pooled_a, _ = self._pool_hidden(x, original_item_idx, n_item)
                pooled_b, counts_b = self._pool_hidden(
                    x, pseudo_item_idx, n_pseudo_item
                )

            residual = None
            if self.use_residual:
                residual = torch.zeros(n_pseudo_item, 1, dtype=x.dtype, device=x.device)
                residual.index_add_(0, pseudo_item_idx, residual_token)
                if self.pooling in ("mean", "mean_pre", "mean_sum"):
                    residual = residual / counts_b.clamp_min(1.0)

            pooled_a = self.norm(pooled_a)
            pooled_b = self.norm(pooled_b)
            a = self.sigmoid(self.linear_a(pooled_a)) * 3 + 1e-6
            b = self.tanh(self.b_head(pooled_b)) * 3
            if self.use_residual:
                b = b + (self.residual_alpha * residual if self.residual_alpha is not None else residual)
            return a, b

    return ItemNetOrdinalSplit


class _CENQBOrdinalSplit:
    def __new__(cls, torch, nn, **kwargs):
        PersonNet = _make_person_net(torch, nn)
        ItemNetOrdinalSplit = _make_item_net_ordinal_split(torch, nn)

        class CENQBOrdinalSplit(nn.Module):
            def __init__(
                self, n_items, n_users, pseudo_to_item, pseudo_step, item_offsets,
                item_max_scores, depth, activation, embedding_dim, hidden_dim,
                pooling, use_residual, use_layernorm, use_b_mlp,
                use_residual_shrinkage, residual_init, irt_model
            ):
                super().__init__()
                self.n_item = n_items
                self.n_person = n_users
                self.n_pseudo_item = int(pseudo_to_item.numel())
                self.irt_model = _normalize_binary_model(irt_model)
                self.person_net = PersonNet(
                    self.n_pseudo_item, depth, activation, hidden_dim, embedding_dim,
                    pooling, use_residual, use_layernorm,
                    use_residual_shrinkage, residual_init
                )
                self.item_net = ItemNetOrdinalSplit(
                    n_users, depth, activation, hidden_dim, embedding_dim, pooling,
                    use_residual, use_layernorm, use_b_mlp,
                    use_residual_shrinkage, residual_init
                )
                self.register_buffer("pseudo_to_item", pseudo_to_item.long())
                self.register_buffer("pseudo_step", pseudo_step.long())
                self.register_buffer("item_offsets", item_offsets.long())
                self.register_buffer("item_max_scores", item_max_scores.long())
                self.register_buffer("_observed_person_idx", torch.empty(0, dtype=torch.long), persistent=False)
                self.register_buffer("_observed_pseudo_item_idx", torch.empty(0, dtype=torch.long), persistent=False)
                self.register_buffer("_observed_response", torch.empty(0, dtype=torch.float32), persistent=False)

            def set_observed_data(self, person_idx, pseudo_item_idx, response):
                device = next(self.parameters()).device
                self._observed_person_idx = person_idx.long().to(device)
                self._observed_pseudo_item_idx = pseudo_item_idx.long().to(device)
                self._observed_response = response.float().to(device)

            def compute_parameters(self):
                if self._observed_response.numel() == 0:
                    raise RuntimeError("Observed ordinal pseudo-response data has not been set.")
                pseudo_to_item = self.pseudo_to_item.to(self._observed_response.device)
                z = self.person_net(
                    self._observed_pseudo_item_idx,
                    self._observed_response,
                    self._observed_person_idx,
                    self.n_person,
                )
                original_item_idx = pseudo_to_item[self._observed_pseudo_item_idx]
                a, b = self.item_net(
                    self._observed_person_idx,
                    self._observed_response,
                    self._observed_pseudo_item_idx,
                    original_item_idx,
                    self.n_item,
                    self.n_pseudo_item,
                )
                if self.irt_model == "1pl":
                    a = torch.ones(self.n_item, 1, dtype=b.dtype, device=b.device)
                return z, a, b

            def forward(self, person_idx, pseudo_item_idx):
                device = self.pseudo_to_item.device
                person_idx = person_idx.long().to(device)
                pseudo_item_idx = pseudo_item_idx.long().to(device)
                z, a, b = self.compute_parameters()
                original_item_idx = self.pseudo_to_item.to(device)[pseudo_item_idx]
                logits = a[original_item_idx] * (z[person_idx] - b[pseudo_item_idx])
                return logits.flatten()

        return CENQBOrdinalSplit(**kwargs)


def _make_person_net_grm(torch, nn):
    class PersonNetGRM(nn.Module):
        def __init__(self, num_item_score_states, depth, activation, hidden_dim, embedding_dim, pooling, use_layernorm):
            super().__init__()
            self.pooling = pooling
            self.item_score_embedding = nn.Embedding(num_item_score_states, embedding_dim)
            mlp_input_dim = embedding_dim * 2 if pooling == "mean_sum" else embedding_dim
            self.layers, self.activation = _make_mlp(nn, mlp_input_dim, hidden_dim, depth, activation)
            self.norm = nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity()
            self.linear = nn.Linear(hidden_dim, 1)
            self.tanh = nn.Tanh()

        def _apply_layers(self, x):
            for layer in self.layers:
                x = self.activation(layer(x))
            return x

        def forward(self, item_score_idx, person_idx, n_person):
            x = self.item_score_embedding(item_score_idx.long().view(-1))
            counts = None
            if self.pooling in ("mean_pre", "mean_sum"):
                pooled_sum = torch.zeros(n_person, x.shape[1], dtype=x.dtype, device=x.device)
                pooled_sum.index_add_(0, person_idx, x)
                counts = torch.zeros(n_person, 1, dtype=x.dtype, device=x.device)
                counts.index_add_(
                    0, person_idx,
                    torch.ones(x.shape[0], 1, dtype=x.dtype, device=x.device)
                )
                pooled_mean = pooled_sum / counts.clamp_min(1.0)
                x = (
                    torch.cat([pooled_mean, pooled_sum], dim=1)
                    if self.pooling == "mean_sum"
                    else pooled_mean
                )
                pooled = self._apply_layers(x)
            else:
                x = self._apply_layers(x)
                pooled = torch.zeros(n_person, x.shape[1], dtype=x.dtype, device=x.device)
                pooled.index_add_(0, person_idx, x)

            if self.pooling == "mean":
                if counts is None:
                    counts = torch.zeros(n_person, 1, dtype=x.dtype, device=x.device)
                    counts.index_add_(
                        0, person_idx,
                        torch.ones(x.shape[0], 1, dtype=x.dtype, device=x.device)
                    )
                pooled = pooled / counts.clamp_min(1.0)
            elif self.pooling not in ("sum", "mean_pre", "mean_sum"):
                raise ValueError(f"Unsupported pooling: {self.pooling}")

            z = self.tanh(self.linear(self.norm(pooled))) * 3
            if z.shape[0] > 1:
                z = (z - z.mean()) / (z.std(unbiased=False) + 1e-6)
            return z

    return PersonNetGRM


def _make_item_net_grm(torch, nn):
    class ItemNetGRM(nn.Module):
        def __init__(self, num_users, max_score, depth, activation, hidden_dim, embedding_dim, pooling, use_layernorm):
            super().__init__()
            self.pooling = pooling
            self.max_score = int(max_score)
            self.person_score_embedding = nn.Embedding(
                num_users * (self.max_score + 1), embedding_dim
            )
            mlp_input_dim = embedding_dim * 2 if pooling == "mean_sum" else embedding_dim
            self.layers, self.activation = _make_mlp(nn, mlp_input_dim, hidden_dim, depth, activation)
            self.norm = nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity()
            self.linear_a = nn.Linear(hidden_dim, 1)
            self.linear_b = nn.Linear(hidden_dim, self.max_score)
            self.sig = nn.Sigmoid()
            self.softplus = nn.Softplus()

        def _apply_layers(self, x):
            for layer in self.layers:
                x = self.activation(layer(x))
            return x

        def forward(self, person_idx, score, item_idx, n_item):
            score_idx = score.long().view(-1)
            if torch.any((score_idx < 0) | (score_idx > self.max_score)):
                raise ValueError("GRM scores must be in 0..max(item_max_scores).")

            embedding_idx = person_idx * (self.max_score + 1) + score_idx
            x = self.person_score_embedding(embedding_idx)
            counts = None
            if self.pooling in ("mean_pre", "mean_sum"):
                pooled_sum = torch.zeros(n_item, x.shape[1], dtype=x.dtype, device=x.device)
                pooled_sum.index_add_(0, item_idx, x)
                counts = torch.zeros(n_item, 1, dtype=x.dtype, device=x.device)
                counts.index_add_(
                    0, item_idx,
                    torch.ones(x.shape[0], 1, dtype=x.dtype, device=x.device)
                )
                pooled_mean = pooled_sum / counts.clamp_min(1.0)
                x = (
                    torch.cat([pooled_mean, pooled_sum], dim=1)
                    if self.pooling == "mean_sum"
                    else pooled_mean
                )
                pooled = self._apply_layers(x)
            else:
                x = self._apply_layers(x)
                pooled = torch.zeros(n_item, x.shape[1], dtype=x.dtype, device=x.device)
                pooled.index_add_(0, item_idx, x)

            if self.pooling == "mean":
                if counts is None:
                    counts = torch.zeros(n_item, 1, dtype=x.dtype, device=x.device)
                    counts.index_add_(
                        0, item_idx,
                        torch.ones(x.shape[0], 1, dtype=x.dtype, device=x.device)
                    )
                pooled = pooled / counts.clamp_min(1.0)
            elif self.pooling not in ("sum", "mean_pre", "mean_sum"):
                raise ValueError(f"Unsupported pooling: {self.pooling}")

            pooled = self.norm(pooled)
            a = self.sig(self.linear_a(pooled)) * 3 + 1e-6
            raw_b = self.linear_b(pooled)
            base_b = torch.tanh(raw_b[:, :1]) * 3
            if self.max_score == 1:
                b = base_b
            else:
                deltas = self.softplus(raw_b[:, 1:]) + 1e-4
                b = torch.cat([base_b, base_b + torch.cumsum(deltas, dim=1)], dim=1)
            return a, b

    return ItemNetGRM


class _CENQBGRM:
    def __new__(cls, torch, nn, **kwargs):
        PersonNetGRM = _make_person_net_grm(torch, nn)
        ItemNetGRM = _make_item_net_grm(torch, nn)

        class CENQBGRM(nn.Module):
            def __init__(
                self, n_items, n_users, item_max_scores, depth,
                activation, embedding_dim, hidden_dim, pooling, use_layernorm,
                irt_model
            ):
                super().__init__()
                item_max_scores = item_max_scores.long().view(-1).cpu()
                if item_max_scores.numel() != n_items:
                    raise ValueError("item_max_scores must have one entry per item.")
                if torch.any(item_max_scores < 1):
                    raise ValueError("GRM item_max_scores must be at least 1 for every item.")

                self.n_item = n_items
                self.n_person = n_users
                self.irt_model = _normalize_binary_model(irt_model)
                self.max_score = int(item_max_scores.max().item())
                item_score_counts = item_max_scores + 1
                item_score_offsets = torch.zeros(n_items + 1, dtype=torch.long)
                item_score_offsets[1:] = torch.cumsum(item_score_counts, dim=0)
                self.register_buffer("item_max_scores", item_max_scores.long())
                self.register_buffer("item_score_offsets", item_score_offsets.long())
                self.person_net = PersonNetGRM(
                    int(item_score_offsets[-1].item()), depth, activation,
                    hidden_dim, embedding_dim, pooling, use_layernorm
                )
                self.item_net = ItemNetGRM(
                    n_users, self.max_score, depth, activation,
                    hidden_dim, embedding_dim, pooling, use_layernorm
                )
                self.register_buffer("_observed_person_idx", torch.empty(0, dtype=torch.long), persistent=False)
                self.register_buffer("_observed_item_idx", torch.empty(0, dtype=torch.long), persistent=False)
                self.register_buffer("_observed_score", torch.empty(0, dtype=torch.long), persistent=False)

            def set_observed_data(self, person_idx, item_idx, score):
                device = next(self.parameters()).device
                self._observed_person_idx = person_idx.long().to(device)
                self._observed_item_idx = item_idx.long().to(device)
                self._observed_score = score.long().to(device)

            def _item_score_idx(self, item_idx, score):
                offsets = self.item_score_offsets.to(item_idx.device)
                return offsets[item_idx.long()] + score.long()

            def compute_parameters(self):
                if self._observed_score.numel() == 0:
                    raise RuntimeError("Observed GRM score data has not been set.")
                item_score_idx = self._item_score_idx(
                    self._observed_item_idx, self._observed_score
                )
                z = self.person_net(
                    item_score_idx,
                    self._observed_person_idx,
                    self.n_person,
                )
                a, b = self.item_net(
                    self._observed_person_idx,
                    self._observed_score,
                    self._observed_item_idx,
                    self.n_item,
                )
                if self.irt_model == "1pl":
                    a = torch.ones(self.n_item, 1, dtype=b.dtype, device=b.device)
                return z, a, b

            def observed_log_prob(self, person_idx, item_idx, score):
                device = self._observed_score.device
                person_idx = person_idx.long().to(device)
                item_idx = item_idx.long().to(device)
                score = score.long().to(device).view(-1)
                z, a, b = self.compute_parameters()
                item_max = self.item_max_scores.to(device)[item_idx]
                if torch.any(score < 0) or torch.any(score > item_max):
                    raise ValueError("Observed GRM scores must be in 0..item_max_scores[item].")

                cum_prob = torch.sigmoid(a[item_idx] * (z[person_idx] - b[item_idx]))
                prob = torch.empty(score.shape[0], dtype=cum_prob.dtype, device=device)
                zero_mask = score == 0
                max_mask = score == item_max
                mid_mask = ~(zero_mask | max_mask)

                if torch.any(zero_mask):
                    prob[zero_mask] = 1 - cum_prob[zero_mask, 0]
                if torch.any(max_mask):
                    last_step_idx = (score[max_mask] - 1).view(-1, 1)
                    prob[max_mask] = cum_prob[max_mask].gather(1, last_step_idx).view(-1)
                if torch.any(mid_mask):
                    left_idx = (score[mid_mask] - 1).view(-1, 1)
                    right_idx = score[mid_mask].view(-1, 1)
                    left = cum_prob[mid_mask].gather(1, left_idx).view(-1)
                    right = cum_prob[mid_mask].gather(1, right_idx).view(-1)
                    prob[mid_mask] = left - right

                return torch.log(prob.clamp_min(1e-8))

            def neg_log_likelihood(self, person_idx, item_idx, score):
                return -self.observed_log_prob(person_idx, item_idx, score).mean()

            def forward(self, person_idx, item_idx, score):
                return self.observed_log_prob(person_idx, item_idx, score)

        return CENQBGRM(**kwargs)


def _make_item_net_multi_ordinal_split(torch, nn):
    class ItemNetMultiOrdinalSplit(nn.Module):
        def __init__(
            self, num_users, dim, depth, activation, hidden_dim, embedding_dim,
            pooling, use_residual, use_layernorm, use_d_mlp,
            use_residual_shrinkage, residual_init, a_upper, d_range
        ):
            super().__init__()
            self.dim = int(dim)
            self.pooling = pooling
            self.use_residual = use_residual
            self.use_layernorm = use_layernorm
            self.use_d_mlp = use_d_mlp
            self.use_residual_shrinkage = use_residual_shrinkage
            self.a_upper = float(a_upper)
            self.d_range = float(d_range)
            self.person_response_embedding = nn.Embedding(num_users * 2, embedding_dim)
            self.person_response_residual = (
                nn.Embedding(num_users * 2, 1) if use_residual else None
            )
            self.residual_alpha = (
                nn.Parameter(torch.tensor(float(residual_init)))
                if use_residual and use_residual_shrinkage
                else None
            )
            mlp_input_dim = embedding_dim * 2 if pooling == "mean_sum" else embedding_dim
            self.layers, self.activation = _make_mlp(
                nn, mlp_input_dim, hidden_dim, depth, activation
            )
            self.norm = nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity()
            self.linear_a = nn.Linear(hidden_dim, self.dim)
            if use_d_mlp:
                d_hidden_dim = max(1, hidden_dim // 2)
                self.d_head = nn.Sequential(
                    nn.Linear(hidden_dim, d_hidden_dim),
                    _activation_layer(nn, activation),
                    nn.Linear(d_hidden_dim, 1),
                )
            else:
                self.d_head = nn.Linear(hidden_dim, 1)
            self.tanh = nn.Tanh()
            self.sig = nn.Sigmoid()

        def _apply_layers(self, x):
            for layer in self.layers:
                x = self.activation(layer(x))
            return x

        def _pool_raw_then_mlp(self, x, group_idx, n_group):
            pooled_sum = torch.zeros(n_group, x.shape[1], dtype=x.dtype, device=x.device)
            pooled_sum.index_add_(0, group_idx, x)
            counts = torch.zeros(n_group, 1, dtype=x.dtype, device=x.device)
            counts.index_add_(
                0, group_idx,
                torch.ones(x.shape[0], 1, dtype=x.dtype, device=x.device)
            )
            pooled_mean = pooled_sum / counts.clamp_min(1.0)
            pooled = (
                torch.cat([pooled_mean, pooled_sum], dim=1)
                if self.pooling == "mean_sum"
                else pooled_mean
            )
            return self._apply_layers(pooled), counts

        def _pool_hidden(self, x, group_idx, n_group):
            pooled = torch.zeros(n_group, x.shape[1], dtype=x.dtype, device=x.device)
            pooled.index_add_(0, group_idx, x)
            counts = torch.zeros(n_group, 1, dtype=x.dtype, device=x.device)
            counts.index_add_(
                0, group_idx,
                torch.ones(x.shape[0], 1, dtype=x.dtype, device=x.device)
            )
            if self.pooling == "mean":
                pooled = pooled / counts.clamp_min(1.0)
            elif self.pooling != "sum":
                raise ValueError(f"Unsupported pooling: {self.pooling}")
            return pooled, counts

        def forward(
            self, person_idx, response, pseudo_item_idx, original_item_idx,
            n_item, n_pseudo_item
        ):
            response_idx = response.long().view(-1)
            if torch.any((response_idx < 0) | (response_idx > 1)):
                raise ValueError(
                    "Neural multi ordinal split expects binary pseudo-responses."
                )

            embedding_idx = person_idx * 2 + response_idx
            x = self.person_response_embedding(embedding_idx)
            residual_token = self.person_response_residual(embedding_idx) if self.use_residual else None

            if self.pooling in ("mean_pre", "mean_sum"):
                pooled_a, _ = self._pool_raw_then_mlp(x, original_item_idx, n_item)
                pooled_d, counts_d = self._pool_raw_then_mlp(
                    x, pseudo_item_idx, n_pseudo_item
                )
            else:
                x = self._apply_layers(x)
                pooled_a, _ = self._pool_hidden(x, original_item_idx, n_item)
                pooled_d, counts_d = self._pool_hidden(
                    x, pseudo_item_idx, n_pseudo_item
                )

            residual = None
            if self.use_residual:
                residual = torch.zeros(n_pseudo_item, 1, dtype=x.dtype, device=x.device)
                residual.index_add_(0, pseudo_item_idx, residual_token)
                if self.pooling in ("mean", "mean_pre", "mean_sum"):
                    residual = residual / counts_d.clamp_min(1.0)

            pooled_a = self.norm(pooled_a)
            pooled_d = self.norm(pooled_d)
            a = self.sig(self.linear_a(pooled_a)) * self.a_upper + 1e-6
            d = self.tanh(self.d_head(pooled_d)) * self.d_range
            if self.use_residual:
                d = d + (
                    self.residual_alpha * residual
                    if self.residual_alpha is not None
                    else residual
                )
            return a, d

    return ItemNetMultiOrdinalSplit


class _CENQBMultiOrdinalSplit:
    def __new__(cls, torch, nn, **kwargs):
        PersonNetMulti = _make_person_net_multi(torch, nn)
        ItemNetMultiOrdinalSplit = _make_item_net_multi_ordinal_split(torch, nn)

        class CENQBMultiOrdinalSplit(nn.Module):
            def __init__(
                self, n_items, n_users, dim, q_matrix, pseudo_to_item,
                pseudo_step, item_offsets, item_max_scores, depth, activation,
                embedding_dim, hidden_dim, pooling, use_residual, use_layernorm,
                use_d_mlp, use_residual_shrinkage, residual_init, a_upper,
                d_range, irt_model="2pl"
            ):
                super().__init__()
                self.n_item = int(n_items)
                self.n_person = int(n_users)
                self.dim = int(dim)
                self.n_pseudo_item = int(pseudo_to_item.numel())
                self.irt_model = _normalize_binary_model(irt_model)
                self.person_net = PersonNetMulti(
                    self.n_pseudo_item, self.dim, depth, activation, hidden_dim,
                    embedding_dim, pooling, use_residual, use_layernorm,
                    use_residual_shrinkage, residual_init
                )
                self.item_net = ItemNetMultiOrdinalSplit(
                    self.n_person, self.dim, depth, activation, hidden_dim,
                    embedding_dim, pooling, use_residual, use_layernorm,
                    use_d_mlp, use_residual_shrinkage, residual_init,
                    a_upper, d_range
                )
                self.register_buffer("q_matrix", q_matrix.float())
                self.register_buffer("pseudo_to_item", pseudo_to_item.long())
                self.register_buffer("pseudo_step", pseudo_step.long())
                self.register_buffer("item_offsets", item_offsets.long())
                self.register_buffer("item_max_scores", item_max_scores.long())
                self.register_buffer("_observed_person_idx", torch.empty(0, dtype=torch.long), persistent=False)
                self.register_buffer("_observed_pseudo_item_idx", torch.empty(0, dtype=torch.long), persistent=False)
                self.register_buffer("_observed_response", torch.empty(0, dtype=torch.float32), persistent=False)

            def set_observed_data(self, person_idx, pseudo_item_idx, response):
                device = next(self.parameters()).device
                self._observed_person_idx = person_idx.long().to(device)
                self._observed_pseudo_item_idx = pseudo_item_idx.long().to(device)
                self._observed_response = response.float().to(device)

            def compute_parameters(self):
                if self._observed_response.numel() == 0:
                    raise RuntimeError(
                        "Observed multi ordinal pseudo-response data has not been set."
                    )
                device = self._observed_response.device
                pseudo_to_item = self.pseudo_to_item.to(device)
                z = self.person_net(
                    self._observed_pseudo_item_idx,
                    self._observed_response,
                    self._observed_person_idx,
                    self.n_person,
                )
                original_item_idx = pseudo_to_item[self._observed_pseudo_item_idx]
                a, d = self.item_net(
                    self._observed_person_idx,
                    self._observed_response,
                    self._observed_pseudo_item_idx,
                    original_item_idx,
                    self.n_item,
                    self.n_pseudo_item,
                )
                q_matrix = self.q_matrix.to(device=d.device, dtype=d.dtype)
                if self.irt_model == "1pl":
                    a = q_matrix
                else:
                    a = a * q_matrix
                return z, a, d

            def forward(self, person_idx, pseudo_item_idx):
                device = self._observed_response.device
                person_idx = person_idx.long().to(device)
                pseudo_item_idx = pseudo_item_idx.long().to(device)
                z, a, d = self.compute_parameters()
                pseudo_to_item = self.pseudo_to_item.to(device)
                original_item_idx = pseudo_to_item[pseudo_item_idx]
                eta = (z[person_idx] * a[original_item_idx]).sum(dim=1)
                logits = eta - d[pseudo_item_idx].view(-1)
                return logits

        return CENQBMultiOrdinalSplit(**kwargs)


def _make_person_net_multi_grm(torch, nn):
    class PersonNetMultiGRM(nn.Module):
        def __init__(
            self, num_item_score_states, dim, depth, activation, hidden_dim,
            embedding_dim, pooling, use_layernorm
        ):
            super().__init__()
            self.dim = int(dim)
            self.pooling = pooling
            self.item_score_embedding = nn.Embedding(num_item_score_states, embedding_dim)
            mlp_input_dim = embedding_dim * 2 if pooling == "mean_sum" else embedding_dim
            self.layers, self.activation = _make_mlp(
                nn, mlp_input_dim, hidden_dim, depth, activation
            )
            self.norm = nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity()
            self.linear = nn.Linear(hidden_dim, self.dim)
            self.tanh = nn.Tanh()

        def _apply_layers(self, x):
            for layer in self.layers:
                x = self.activation(layer(x))
            return x

        def forward(self, item_score_idx, person_idx, n_person):
            x = self.item_score_embedding(item_score_idx.long().view(-1))
            counts = None
            if self.pooling in ("mean_pre", "mean_sum"):
                pooled_sum = torch.zeros(n_person, x.shape[1], dtype=x.dtype, device=x.device)
                pooled_sum.index_add_(0, person_idx, x)
                counts = torch.zeros(n_person, 1, dtype=x.dtype, device=x.device)
                counts.index_add_(
                    0, person_idx,
                    torch.ones(x.shape[0], 1, dtype=x.dtype, device=x.device)
                )
                pooled_mean = pooled_sum / counts.clamp_min(1.0)
                x = (
                    torch.cat([pooled_mean, pooled_sum], dim=1)
                    if self.pooling == "mean_sum"
                    else pooled_mean
                )
                pooled = self._apply_layers(x)
            else:
                x = self._apply_layers(x)
                pooled = torch.zeros(n_person, x.shape[1], dtype=x.dtype, device=x.device)
                pooled.index_add_(0, person_idx, x)

            if self.pooling == "mean":
                if counts is None:
                    counts = torch.zeros(n_person, 1, dtype=x.dtype, device=x.device)
                    counts.index_add_(
                        0, person_idx,
                        torch.ones(x.shape[0], 1, dtype=x.dtype, device=x.device)
                    )
                pooled = pooled / counts.clamp_min(1.0)
            elif self.pooling not in ("sum", "mean_pre", "mean_sum"):
                raise ValueError(f"Unsupported pooling: {self.pooling}")

            pooled = self.norm(pooled)
            z = self.tanh(self.linear(pooled)) * 3
            if z.shape[0] > 1:
                z = (z - z.mean(dim=0, keepdim=True)) / (
                    z.std(dim=0, unbiased=False, keepdim=True) + 1e-6
                )
            return z

    return PersonNetMultiGRM


def _make_item_net_multi_grm(torch, nn):
    class ItemNetMultiGRM(nn.Module):
        def __init__(
            self, num_users, dim, max_score, depth, activation, hidden_dim,
            embedding_dim, pooling, use_layernorm, a_upper, d_range
        ):
            super().__init__()
            self.dim = int(dim)
            self.pooling = pooling
            self.max_score = int(max_score)
            self.a_upper = float(a_upper)
            self.d_range = float(d_range)
            self.person_score_embedding = nn.Embedding(
                num_users * (self.max_score + 1), embedding_dim
            )
            mlp_input_dim = embedding_dim * 2 if pooling == "mean_sum" else embedding_dim
            self.layers, self.activation = _make_mlp(
                nn, mlp_input_dim, hidden_dim, depth, activation
            )
            self.norm = nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity()
            self.linear_a = nn.Linear(hidden_dim, self.dim)
            self.linear_d = nn.Linear(hidden_dim, self.max_score)
            self.sig = nn.Sigmoid()
            self.softplus = nn.Softplus()

        def _apply_layers(self, x):
            for layer in self.layers:
                x = self.activation(layer(x))
            return x

        def forward(self, person_idx, score, item_idx, n_item):
            score_idx = score.long().view(-1)
            if torch.any((score_idx < 0) | (score_idx > self.max_score)):
                raise ValueError("MultiGRM scores must be in 0..max(item_max_scores).")

            embedding_idx = person_idx * (self.max_score + 1) + score_idx
            x = self.person_score_embedding(embedding_idx)
            counts = None
            if self.pooling in ("mean_pre", "mean_sum"):
                pooled_sum = torch.zeros(n_item, x.shape[1], dtype=x.dtype, device=x.device)
                pooled_sum.index_add_(0, item_idx, x)
                counts = torch.zeros(n_item, 1, dtype=x.dtype, device=x.device)
                counts.index_add_(
                    0, item_idx,
                    torch.ones(x.shape[0], 1, dtype=x.dtype, device=x.device)
                )
                pooled_mean = pooled_sum / counts.clamp_min(1.0)
                x = (
                    torch.cat([pooled_mean, pooled_sum], dim=1)
                    if self.pooling == "mean_sum"
                    else pooled_mean
                )
                pooled = self._apply_layers(x)
            else:
                x = self._apply_layers(x)
                pooled = torch.zeros(n_item, x.shape[1], dtype=x.dtype, device=x.device)
                pooled.index_add_(0, item_idx, x)

            if self.pooling == "mean":
                if counts is None:
                    counts = torch.zeros(n_item, 1, dtype=x.dtype, device=x.device)
                    counts.index_add_(
                        0, item_idx,
                        torch.ones(x.shape[0], 1, dtype=x.dtype, device=x.device)
                    )
                pooled = pooled / counts.clamp_min(1.0)
            elif self.pooling not in ("sum", "mean_pre", "mean_sum"):
                raise ValueError(f"Unsupported pooling: {self.pooling}")

            pooled = self.norm(pooled)
            a = self.sig(self.linear_a(pooled)) * self.a_upper + 1e-6
            raw_d = self.linear_d(pooled)
            base_d = torch.tanh(raw_d[:, :1]) * self.d_range
            if self.max_score == 1:
                d = base_d
            else:
                deltas = self.softplus(raw_d[:, 1:]) + 1e-4
                d = torch.cat([base_d, base_d + torch.cumsum(deltas, dim=1)], dim=1)
            return a, d

    return ItemNetMultiGRM


class _CENQBMultiGRM:
    def __new__(cls, torch, nn, **kwargs):
        PersonNetMultiGRM = _make_person_net_multi_grm(torch, nn)
        ItemNetMultiGRM = _make_item_net_multi_grm(torch, nn)

        class CENQBMultiGRM(nn.Module):
            def __init__(
                self, n_items, n_users, dim, q_matrix, item_max_scores, depth,
                activation, embedding_dim, hidden_dim, pooling, use_layernorm,
                a_upper, d_range, irt_model="2pl"
            ):
                super().__init__()
                item_max_scores = item_max_scores.long().view(-1).cpu()
                if item_max_scores.numel() != n_items:
                    raise ValueError("item_max_scores must have one entry per item.")
                if torch.any(item_max_scores < 1):
                    raise ValueError("MultiGRM item_max_scores must be at least 1 for every item.")

                self.n_item = int(n_items)
                self.n_person = int(n_users)
                self.dim = int(dim)
                self.irt_model = _normalize_binary_model(irt_model)
                self.max_score = int(item_max_scores.max().item())
                item_score_counts = item_max_scores + 1
                item_score_offsets = torch.zeros(self.n_item + 1, dtype=torch.long)
                item_score_offsets[1:] = torch.cumsum(item_score_counts, dim=0)

                self.register_buffer("q_matrix", q_matrix.float())
                self.register_buffer("item_max_scores", item_max_scores.long())
                self.register_buffer("item_score_offsets", item_score_offsets.long())
                self.person_net = PersonNetMultiGRM(
                    int(item_score_offsets[-1].item()), self.dim, depth, activation,
                    hidden_dim, embedding_dim, pooling, use_layernorm
                )
                self.item_net = ItemNetMultiGRM(
                    self.n_person, self.dim, self.max_score, depth, activation,
                    hidden_dim, embedding_dim, pooling, use_layernorm,
                    a_upper, d_range
                )
                self.register_buffer("_observed_person_idx", torch.empty(0, dtype=torch.long), persistent=False)
                self.register_buffer("_observed_item_idx", torch.empty(0, dtype=torch.long), persistent=False)
                self.register_buffer("_observed_score", torch.empty(0, dtype=torch.long), persistent=False)

            def set_observed_data(self, person_idx, item_idx, score):
                device = next(self.parameters()).device
                self._observed_person_idx = person_idx.long().to(device)
                self._observed_item_idx = item_idx.long().to(device)
                self._observed_score = score.long().to(device)

            def _item_score_idx(self, item_idx, score):
                offsets = self.item_score_offsets.to(item_idx.device)
                return offsets[item_idx.long()] + score.long()

            def compute_parameters(self):
                if self._observed_score.numel() == 0:
                    raise RuntimeError("Observed MultiGRM score data has not been set.")
                item_score_idx = self._item_score_idx(
                    self._observed_item_idx, self._observed_score
                )
                z = self.person_net(
                    item_score_idx,
                    self._observed_person_idx,
                    self.n_person,
                )
                a, d = self.item_net(
                    self._observed_person_idx,
                    self._observed_score,
                    self._observed_item_idx,
                    self.n_item,
                )
                q_matrix = self.q_matrix.to(device=d.device, dtype=d.dtype)
                if self.irt_model == "1pl":
                    a = q_matrix
                else:
                    a = a * q_matrix
                return z, a, d

            def cumulative_logits(self, person_idx, item_idx):
                device = self._observed_score.device
                person_idx = person_idx.long().to(device)
                item_idx = item_idx.long().to(device)
                z, a, d = self.compute_parameters()
                eta = (z[person_idx] * a[item_idx]).sum(dim=1, keepdim=True)
                logits = eta - d[item_idx]
                steps = torch.arange(1, self.max_score + 1, device=device).view(1, -1)
                valid = steps <= self.item_max_scores.to(device)[item_idx].view(-1, 1)
                return logits.masked_fill(~valid, float("nan"))

            def observed_log_prob(self, person_idx, item_idx, score):
                device = self._observed_score.device
                person_idx = person_idx.long().to(device)
                item_idx = item_idx.long().to(device)
                score = score.long().to(device).view(-1)
                z, a, d = self.compute_parameters()

                item_max = self.item_max_scores.to(device)[item_idx]
                if torch.any(score < 0) or torch.any(score > item_max):
                    raise ValueError(
                        "Observed MultiGRM scores must be in 0..item_max_scores[item]."
                    )

                eta = (z[person_idx] * a[item_idx]).sum(dim=1, keepdim=True)
                cum_prob = torch.sigmoid(eta - d[item_idx])
                prob = torch.empty(score.shape[0], dtype=cum_prob.dtype, device=device)

                zero_mask = score == 0
                max_mask = score == item_max
                mid_mask = ~(zero_mask | max_mask)

                if torch.any(zero_mask):
                    prob[zero_mask] = 1 - cum_prob[zero_mask, 0]
                if torch.any(max_mask):
                    last_step_idx = (score[max_mask] - 1).view(-1, 1)
                    prob[max_mask] = cum_prob[max_mask].gather(1, last_step_idx).view(-1)
                if torch.any(mid_mask):
                    left_idx = (score[mid_mask] - 1).view(-1, 1)
                    right_idx = score[mid_mask].view(-1, 1)
                    left = cum_prob[mid_mask].gather(1, left_idx).view(-1)
                    right = cum_prob[mid_mask].gather(1, right_idx).view(-1)
                    prob[mid_mask] = left - right

                return torch.log(prob.clamp_min(1e-8))

            def neg_log_likelihood(self, person_idx, item_idx, score):
                return -self.observed_log_prob(person_idx, item_idx, score).mean()

            def forward(self, person_idx, item_idx, score):
                return self.observed_log_prob(person_idx, item_idx, score)

        return CENQBMultiGRM(**kwargs)
