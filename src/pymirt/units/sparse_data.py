from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SparseResponse:
    """Internal sparse response store based on observed cells only."""

    user_idx: np.ndarray
    item_idx: np.ndarray
    values: np.ndarray
    n_users: int
    n_items: int
    user_order: np.ndarray
    user_indptr: np.ndarray
    item_order: np.ndarray
    item_indptr: np.ndarray

    @classmethod
    def from_arrays(cls, user_idx, item_idx, values, n_users, n_items):
        user_idx = np.asarray(user_idx, dtype=np.int64)
        item_idx = np.asarray(item_idx, dtype=np.int64)
        values = np.asarray(values, dtype=float)

        if user_idx.shape != item_idx.shape or user_idx.shape != values.shape:
            raise ValueError("Sparse response arrays must have matching 1-D shapes.")
        if n_users < 1 or n_items < 1:
            raise ValueError("Sparse response shape must be non-empty.")
        if len(values) == 0:
            raise ValueError("No observed responses were found.")
        if np.any(user_idx < 0) or np.any(user_idx >= n_users):
            raise ValueError("Sparse response contains out-of-range user indexes.")
        if np.any(item_idx < 0) or np.any(item_idx >= n_items):
            raise ValueError("Sparse response contains out-of-range item indexes.")

        user_order, user_indptr = _group_order(user_idx, n_users)
        item_order, item_indptr = _group_order(item_idx, n_items)
        return cls(
            user_idx=user_idx,
            item_idx=item_idx,
            values=values,
            n_users=int(n_users),
            n_items=int(n_items),
            user_order=user_order,
            user_indptr=user_indptr,
            item_order=item_order,
            item_indptr=item_indptr,
        )

    @property
    def n_obs(self):
        return int(self.values.size)

    @property
    def by_user(self):
        return self.user_order, self.user_indptr

    @property
    def by_item(self):
        return self.item_order, self.item_indptr

    def user_observations(self, user_id):
        start, end = self.user_indptr[user_id], self.user_indptr[user_id + 1]
        return self.user_order[start:end]

    def item_observations(self, item_id):
        start, end = self.item_indptr[item_id], self.item_indptr[item_id + 1]
        return self.item_order[start:end]

    def with_values(self, values):
        return SparseResponse.from_arrays(
            self.user_idx,
            self.item_idx,
            values,
            self.n_users,
            self.n_items,
        )

    def subset_items(self, item_mask, values=None):
        item_mask = np.asarray(item_mask, dtype=bool)
        if item_mask.shape != (self.n_items,):
            raise ValueError("item_mask length must match n_items.")

        keep = item_mask[self.item_idx]
        old_to_new = np.full(self.n_items, -1, dtype=np.int64)
        old_to_new[item_mask] = np.arange(np.sum(item_mask), dtype=np.int64)
        new_values = self.values if values is None else np.asarray(values, dtype=float)

        return SparseResponse.from_arrays(
            self.user_idx[keep],
            old_to_new[self.item_idx[keep]],
            new_values[keep],
            self.n_users,
            int(np.sum(item_mask)),
        )


def dataframe_to_sparse_response(response_df):
    """Build a SparseResponse from a wide response DataFrame with NaN missing."""

    if response_df.empty:
        raise ValueError("response_df must not be empty.")

    frame = response_df.copy(deep=False)
    frame.index = pd.RangeIndex(len(frame.index))
    frame.columns = pd.RangeIndex(len(frame.columns))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        stacked = frame.stack(dropna=True)

    if stacked.empty:
        raise ValueError("response_df contains no observed responses.")

    user_idx = stacked.index.get_level_values(0).to_numpy(dtype=np.int64)
    item_idx = stacked.index.get_level_values(1).to_numpy(dtype=np.int64)
    values = stacked.to_numpy(dtype=float)

    return SparseResponse.from_arrays(
        user_idx=user_idx,
        item_idx=item_idx,
        values=values,
        n_users=response_df.shape[0],
        n_items=response_df.shape[1],
    )


def analyze_sparse_response(sparse_response, n_categories=None, binary=False):
    """Validate sparse observed responses without materializing dense matrices."""

    values = sparse_response.values
    if not np.all(np.isfinite(values)):
        raise ValueError("Observed responses must be finite numeric values.")

    item_counts = np.bincount(
        sparse_response.item_idx, minlength=sparse_response.n_items
    )
    empty_items = np.flatnonzero(item_counts == 0)
    if empty_items.size:
        raise ValueError(
            "Items with no observed responses are not supported: "
            + ", ".join(map(str, empty_items.tolist()))
        )

    if binary:
        valid = np.isin(values, [0.0, 1.0])
        if not np.all(valid):
            raise ValueError("Binary IRT models require observed values to be 0 or 1.")

    if n_categories is not None:
        n_categories = np.asarray(n_categories)
        if n_categories.ndim != 1 or len(n_categories) != sparse_response.n_items:
            raise ValueError("n_categories must be a 1-D array with one entry per item.")
        if np.any(n_categories < 2):
            raise ValueError("Each item must have at least two categories.")

        int_values = values.astype(int)
        if not np.allclose(values, int_values):
            raise ValueError("GRM/MGRM observed values must be integer categories.")
        upper = n_categories[sparse_response.item_idx]
        if np.any(int_values < 0) or np.any(int_values >= upper):
            raise ValueError("Observed category values are outside n_categories bounds.")

    return item_counts


def _group_order(indexes, n_groups):
    order = np.argsort(indexes, kind="mergesort")
    counts = np.bincount(indexes, minlength=n_groups)
    indptr = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
    return order.astype(np.int64), indptr
