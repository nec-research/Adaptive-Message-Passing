from random import shuffle

import numpy as np
import pydgn
import torch
from numpy import random
from pydgn.data.splitter import Splitter, OuterFold, InnerFold


class DebugSplitter(Splitter):
    def split(
        self,
        dataset: pydgn.data.dataset.DatasetInterface,
        targets: np.ndarray = None,
    ):
        r"""
        Computes the splits and stores them in the list fields
        ``self.outer_folds`` and ``self.inner_folds``.
        IMPORTANT: calling split() sets the seed of numpy, torch, and
        random for reproducibility.

        Args:
            dataset (:class:`~pydgn.data.dataset.DatasetInterface`):
                the Dataset object
            targets (np.ndarray]): targets used for stratification.
                Default is ``None``
        """
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        random.seed(self.seed)

        idxs = range(len(dataset))

        stratified = self.stratify
        outer_idxs = np.array(idxs)

        outer_splitter = self._get_splitter(
            n_splits=self.n_outer_folds,
            stratified=stratified,
            eval_ratio=self.test_ratio,
        )  # This is the true test (outer test)

        for train_idxs, test_idxs in outer_splitter.split(
            outer_idxs, y=targets if stratified else None
        ):
            assert set(train_idxs) == set(outer_idxs[train_idxs])
            assert set(test_idxs) == set(outer_idxs[test_idxs])

            inner_fold_splits = []
            inner_idxs = outer_idxs[
                train_idxs
            ]  # equals train_idxs because outer_idxs was ordered
            inner_targets = (
                targets[train_idxs] if targets is not None else None
            )

            inner_splitter = self._get_splitter(
                n_splits=self.n_inner_folds,
                stratified=stratified,
                eval_ratio=self.inner_val_ratio,
            )  # The inner "test" is, instead, the validation set

            for inner_train_idxs, inner_val_idxs in inner_splitter.split(
                inner_idxs, y=inner_targets if stratified else None
            ):
                if self.inner_val_ratio == 0.0:
                    inner_fold = InnerFold(
                        train_idxs=inner_idxs[inner_train_idxs].tolist(),
                        val_idxs=inner_idxs[inner_train_idxs[:2]].tolist(),
                    )
                else:
                    inner_fold = InnerFold(
                        train_idxs=inner_idxs[inner_train_idxs].tolist(),
                        val_idxs=inner_idxs[inner_val_idxs].tolist(),
                    )
                inner_fold_splits.append(inner_fold)

                # False if empty
                assert not bool(
                    set(inner_train_idxs)
                    & set(inner_val_idxs)
                    & set(test_idxs)
                )
                assert not bool(
                    set(inner_idxs[inner_train_idxs])
                    & set(inner_idxs[inner_val_idxs])
                    & set(test_idxs)
                )

            self.inner_folds.append(inner_fold_splits)

            # Obtain outer val from outer train in an holdout fashion
            outer_val_splitter = self._get_splitter(
                n_splits=1,
                stratified=stratified,
                eval_ratio=self.outer_val_ratio,
            )

            outer_train_idxs, outer_val_idxs = list(
                outer_val_splitter.split(inner_idxs, y=inner_targets)
            )[0]

            # False if empty
            assert not bool(
                set(outer_train_idxs) & set(outer_val_idxs) & set(test_idxs)
            )
            assert not bool(
                set(outer_train_idxs) & set(outer_val_idxs) & set(test_idxs)
            )
            assert not bool(
                set(inner_idxs[outer_train_idxs])
                & set(inner_idxs[outer_val_idxs])
                & set(test_idxs)
            )

            np.random.shuffle(outer_train_idxs)
            np.random.shuffle(outer_val_idxs)
            np.random.shuffle(test_idxs)
            if self.outer_val_ratio == 0.0:
                outer_fold = OuterFold(
                    train_idxs=inner_idxs[outer_train_idxs].tolist(),
                    val_idxs=inner_idxs[outer_train_idxs[:2]].tolist(),
                    test_idxs=outer_idxs[test_idxs].tolist(),
                )
            else:
                outer_fold = OuterFold(
                    train_idxs=inner_idxs[outer_train_idxs].tolist(),
                    val_idxs=inner_idxs[outer_val_idxs].tolist(),
                    test_idxs=outer_idxs[test_idxs].tolist(),
                )
            self.outer_folds.append(outer_fold)


class GraphPropPredSplitter(Splitter):
    def split(
        self,
        dataset: pydgn.data.dataset.DatasetInterface,
        targets: np.ndarray = None,
    ):
        r"""
        Computes the splits and stores them in the list fields
        ``self.outer_folds`` and ``self.inner_folds``.
        IMPORTANT: calling split() sets the seed of numpy, torch, and
        random for reproducibility.

        Args:
            dataset (:class:`~pydgn.data.dataset.DatasetInterface`):
                the Dataset object
            targets (np.ndarray]): targets used for stratification.
                Default is ``None``
        """
        assert len(dataset) == 5120+640+1280
        assert self.n_inner_folds == 1
        self.n_outer_folds == 1

        train_idxs = torch.arange(0,5120)
        val_idxs = torch.arange(5120, 5120+640)
        test_idxs = torch.arange(5120+640, 5120+640+1280)

        inner_fold_splits = []

        inner_fold = InnerFold(
            train_idxs=train_idxs.tolist(),
            val_idxs=val_idxs.tolist(),
        )
        inner_fold_splits.append(inner_fold)

        self.inner_folds.append(inner_fold_splits)

        outer_fold = OuterFold(
            train_idxs=train_idxs.tolist(),
            val_idxs=val_idxs.tolist(),
            test_idxs=test_idxs.tolist(),
        )

        self.outer_folds.append(outer_fold)


class PeptidesSplitter(Splitter):
    # PyG version
    def split(
        self,
        dataset: pydgn.data.dataset.DatasetInterface,
        targets: np.ndarray = None,
    ):
        r"""
        Computes the splits and stores them in the list fields
        ``self.outer_folds`` and ``self.inner_folds``.
        IMPORTANT: calling split() sets the seed of numpy, torch, and
        random for reproducibility.

        Args:
            dataset (:class:`~pydgn.data.dataset.DatasetInterface`):
                the Dataset object
            targets (np.ndarray]): targets used for stratification.
                Default is ``None``
        """
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        random.seed(self.seed)

        idxs = list(range(len(dataset)))

        assert len(dataset) == 10873+2331+2331
        assert self.n_inner_folds == 1
        self.n_outer_folds == 1

        train_idxs = idxs[:10873]
        val_idxs = idxs[10873:10873+2331]
        test_idxs = idxs[10873+2331:10873+2331+2331]

        assert len(train_idxs) == 10873
        assert len(val_idxs) == 2331
        assert len(test_idxs) == 2331
        assert len(train_idxs) + len(val_idxs) + len(test_idxs) == len(dataset)

        assert set(train_idxs).isdisjoint(set(val_idxs)), "Sets overlap"
        assert set(train_idxs).isdisjoint(set(test_idxs)), "Sets overlap"
        assert set(val_idxs).isdisjoint(set(test_idxs)), "Sets overlap"


        inner_fold_splits = []

        inner_fold = InnerFold(
            train_idxs=train_idxs,
            val_idxs=val_idxs,
        )
        inner_fold_splits.append(inner_fold)

        self.inner_folds.append(inner_fold_splits)

        outer_fold = OuterFold(
            train_idxs=train_idxs,
            val_idxs=val_idxs,
            test_idxs=test_idxs,
        )

        self.outer_folds.append(outer_fold)
