import math
from typing import Any, Dict, Iterable, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np


class Analysis:
    """
    PCA analysis helper.

    - Pre-configured aliases map to source names, PCA modes, and storage destinations.
    - Per-epoch PCA metrics: eigenvalues, explained/cumulative variance,
      k95, effective dimension, spectral sharpness, top-k cumulative variance,
      and principal components.
    - Visualization supports online/offline modes and three views:
      train-valid, epoch, and combined.
    """

    VALID_MODES = {"token-pca", "sample-pca", "per-sample token-pca"}
    VALID_DESTS = {"train-probe", "valid-probe", "epoch-probe"}
    VALID_VIEWS = {"train-valid", "epoch", "combined"}

    def __init__(
        self,
        pca_modes: Optional[Iterable[str]] = None,
        visualize_mode: str = "offline",
        plot_view: str = "combined",
        top_k: int = 2,
        spectrum_interval: int = 5,
        component_interval: int = 5,
    ) -> None:
        self.set_pca_modes(pca_modes or [])
        self.visualize_mode = visualize_mode  # "online" or "offline"
        self.plot_view = plot_view  # "train-valid", "epoch", "combined"
        self.top_k = max(1, int(top_k))
        self.spectrum_interval = max(1, int(spectrum_interval))
        self.component_interval = max(1, int(component_interval))
        self.default_metrics = [
            "effective_dimension",
            "k95",
            "spectral_sharpness",
            "topk_cumvar",
        ]

        self.target_config: Dict[str, Dict[str, Any]] = {}
        self.plot_config: Dict[str, Dict[str, Any]] = {"train-valid": {}, "epoch": {}}
        self.records: Dict[str, Dict[str, Dict[str, List[Dict[str, Any]]]]] = {
            "train-probe": {},
            "valid-probe": {},
            "epoch-probe": {},
        }

    def set_pca_modes(self, modes: Iterable[str]) -> None:
        """Set default PCA modes (may coexist)."""
        normalized = set()
        for m in modes:
            if m not in self.VALID_MODES:
                raise ValueError(f"Invalid PCA mode: {m}")
            normalized.add(m)
        self.pca_modes = normalized or {"sample-pca"}

    def configure_targets(self, target_config: Dict[str, Dict[str, Any]]) -> None:
        """
        Configure alias -> source name, destination, PCA modes.

        target_config example:
        {
            "tokens_train": {
                "source": "token_states",
                "destination": "train-probe",
                "pca_modes": ["token-pca"],
                "label": "train",
            },
            "tokens_valid": {
                "source": "valid_token_states",
                "destination": "valid-probe",
                "pca_modes": ["token-pca"],
                "label": "valid",
            },
            "epoch_repr": {
                "source": "epoch_repr",
                "destination": "epoch-probe",
                "pca_modes": ["sample-pca"],
            },
        }
        """
        for alias, cfg in target_config.items():
            if "source" not in cfg or "destination" not in cfg:
                raise ValueError(f"Alias {alias} requires 'source' and 'destination'")
            dest = cfg["destination"]
            if dest not in self.VALID_DESTS:
                raise ValueError(f"Invalid destination for {alias}: {dest}")
            modes = cfg.get("pca_modes")
            if modes is not None:
                for m in modes:
                    if m not in self.VALID_MODES:
                        raise ValueError(f"Invalid PCA mode for {alias}: {m}")
            self.records[dest].setdefault(alias, {})
        self.target_config = target_config

    def configure_visualization(
        self,
        mode: str = "offline",
        plot_view: Optional[str] = None,
        spectrum_interval: Optional[int] = None,
        component_interval: Optional[int] = None,
    ) -> None:
        """Update visualization settings."""
        if mode not in {"online", "offline"}:
            raise ValueError("Visualization mode must be 'online' or 'offline'")
        self.visualize_mode = mode
        if plot_view is not None:
            if plot_view not in self.VALID_VIEWS:
                raise ValueError("plot_view must be 'train-valid', 'epoch', or 'combined'")
            self.plot_view = plot_view
        if spectrum_interval is not None:
            self.spectrum_interval = max(1, int(spectrum_interval))
        if component_interval is not None:
            self.component_interval = max(1, int(component_interval))

    def configure_plots(self, plot_config: Dict[str, Dict[str, Any]]) -> None:
        """
        Configure visualization groups.

        plot_config example:
        {
            "train-valid": {
                "groups": {
                    "tokens": {
                        "train": ["tokens_train"],
                        "valid": ["tokens_valid"],
                        "metrics": ["k95", "effective_dimension"],
                        "modes": ["token-pca"],
                        "plot_spectrum": True,
                        "plot_components": False,
                    }
                }
            },
            "epoch": {
                "groups": {
                    "epoch_repr": {
                        "aliases": ["epoch_repr"],
                        "metrics": ["k95", "spectral_sharpness"],
                        "modes": ["sample-pca"],
                        "plot_spectrum": False,
                        "plot_components": False,
                    }
                }
            },
        }
        """
        self.plot_config = {"train-valid": {}, "epoch": {}}
        for view, cfg in plot_config.items():
            if view not in {"train-valid", "epoch"}:
                raise ValueError(f"Invalid plot view: {view}")
            groups = cfg.get("groups", {})
            if not isinstance(groups, dict):
                raise ValueError(f"Plot groups for {view} must be a dict")
            for group_name, gcfg in groups.items():
                if view == "train-valid":
                    if "train" not in gcfg or "valid" not in gcfg:
                        raise ValueError(f"Group {group_name} requires 'train' and 'valid' aliases")
                    self._validate_aliases(gcfg["train"])
                    self._validate_aliases(gcfg["valid"])
                else:
                    if "aliases" not in gcfg:
                        raise ValueError(f"Group {group_name} requires 'aliases'")
                    self._validate_aliases(gcfg["aliases"])
                self._validate_modes(gcfg.get("modes"))
            self.plot_config[view] = cfg

    def _validate_aliases(self, aliases: Iterable[str]) -> None:
        for alias in aliases:
            if alias not in self.target_config:
                raise ValueError(f"Alias {alias} not configured")

    def _validate_modes(self, modes: Optional[Iterable[str]]) -> None:
        if modes is None:
            return
        for m in modes:
            if m not in self.VALID_MODES:
                raise ValueError(f"Invalid PCA mode: {m}")

    def process_epoch(self, payload: Dict[str, np.ndarray], epoch: int, strict: bool = False) -> None:
        """
        Process all configured aliases by looking up their source name from `payload`.
        If strict=True, raise when a configured source name is missing.
        """
        for alias, cfg in self.target_config.items():
            source = cfg["source"]
            if source not in payload:
                if strict:
                    raise KeyError(f"Missing source '{source}' for alias '{alias}'")
                continue
            self.process(alias, payload[source], epoch)

    def process(self, alias: str, data: np.ndarray, epoch: int) -> None:
        """
        Run PCA analysis for a given alias and store metrics.
        `data` can be shaped:
        - (n_samples, dim) for sample-pca
        - (n_samples, tokens, dim) for token-related analyses
        """
        if alias not in self.target_config:
            raise KeyError(f"Alias {alias} not configured.")
        dest = self.target_config[alias]["destination"]

        metrics_by_mode: Dict[str, Dict[str, Any]] = {}
        modes = self.target_config[alias].get("pca_modes", self.pca_modes)
        for mode in modes:
            metrics_by_mode[mode] = self._compute_mode_metrics(mode, data)

        store = self.records[dest].setdefault(alias, {})
        for mode, metrics in metrics_by_mode.items():
            store.setdefault(mode, []).append({"epoch": epoch, **metrics})

        if self.visualize_mode == "online":
            self.plot_all(show=False)

    def _compute_mode_metrics(self, mode: str, data: np.ndarray) -> Dict[str, Any]:
        if mode == "sample-pca":
            matrix = self._as_sample_matrix(data)
            return self._pca_metrics(matrix)
        if mode == "token-pca":
            matrix = self._as_token_matrix(data)
            return self._pca_metrics(matrix)
        if mode == "per-sample token-pca":
            return self._per_sample_token_metrics(data)
        raise ValueError(f"Unhandled PCA mode: {mode}")

    def _as_sample_matrix(self, data: np.ndarray) -> np.ndarray:
        """Ensure 2D (samples x dim); if tokens are present, average over tokens."""
        if data.ndim == 2:
            return data
        if data.ndim == 3:
            return data.mean(axis=1)
        raise ValueError(f"Unsupported data shape for sample-pca: {data.shape}")

    def _as_token_matrix(self, data: np.ndarray) -> np.ndarray:
        """Flatten samples and tokens into rows for token-level PCA."""
        if data.ndim == 2:
            return data
        if data.ndim == 3:
            n, t, d = data.shape
            return data.reshape(n * t, d)
        raise ValueError(f"Unsupported data shape for token-pca: {data.shape}")

    def _per_sample_token_metrics(self, data: np.ndarray) -> Dict[str, Any]:
        """Compute token-level PCA per sample and aggregate statistics."""
        if data.ndim != 3:
            raise ValueError(f"per-sample token-pca requires 3D data, got {data.shape}")
        per_sample = [self._pca_metrics(sample) for sample in data]

        def _scalar_stats(key: str) -> Dict[str, float]:
            values = np.array([m[key] for m in per_sample], dtype=float)
            return {
                f"{key}_mean": float(values.mean()),
                f"{key}_std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            }

        eigen_list = [m["eigenvalues"] for m in per_sample]
        expl_list = [m["explained_variance_ratio"] for m in per_sample]
        cum_list = [m["cumulative_variance_ratio"] for m in per_sample]
        eig_mean = self._nanmean_stacked(eigen_list)
        expl_mean = self._nanmean_stacked(expl_list)
        cum_mean = self._nanmean_stacked(cum_list)

        aggregated: Dict[str, Any] = {
            "eigenvalues": eig_mean,
            "explained_variance_ratio": expl_mean,
            "cumulative_variance_ratio": cum_mean,
            "samples": len(per_sample),
        }
        aggregated.update(_scalar_stats("k95"))
        aggregated.update(_scalar_stats("effective_dimension"))
        aggregated.update(_scalar_stats("spectral_sharpness"))
        aggregated.update(_scalar_stats("topk_cumvar"))

        return aggregated

    def _pca_metrics(self, matrix: np.ndarray) -> Dict[str, Any]:
        """Compute PCA metrics on a 2D matrix (rows are observations)."""
        if matrix.ndim != 2:
            raise ValueError(f"PCA expects 2D matrix, got {matrix.shape}")
        if matrix.shape[0] < 2:
            raise ValueError("Need at least 2 samples for PCA.")

        centered = matrix - matrix.mean(axis=0, keepdims=True)
        n_samples = centered.shape[0]
        _, s, vt = np.linalg.svd(centered, full_matrices=False)

        eigenvalues = (s ** 2) / max(n_samples - 1, 1)
        total_var = float(eigenvalues.sum())
        eps = 1e-12
        explained = eigenvalues / (total_var + eps)
        cumulative = np.cumsum(explained)

        k95 = int(np.searchsorted(cumulative, 0.95) + 1) if total_var > 0 else 0
        effective_dim = (total_var ** 2) / (float(np.sum(eigenvalues ** 2)) + eps)
        mean_var = total_var / max(len(eigenvalues), 1)
        spectral_sharpness = float(eigenvalues[0] / (mean_var + eps)) if len(eigenvalues) else 0.0
        topk_idx = min(self.top_k, len(cumulative)) - 1
        topk_cumvar = float(cumulative[topk_idx]) if len(cumulative) else 0.0

        components = vt[: self.top_k] if vt is not None else None

        return {
            "eigenvalues": eigenvalues,
            "explained_variance_ratio": explained,
            "cumulative_variance_ratio": cumulative,
            "k95": k95,
            "effective_dimension": float(effective_dim),
            "spectral_sharpness": spectral_sharpness,
            "topk_cumvar": topk_cumvar,
            "components": components,
        }

    @staticmethod
    def _nanmean_stacked(arrays: List[np.ndarray]) -> np.ndarray:
        """Pad variable-length 1D arrays with NaN then compute nanmean."""
        max_len = max(len(a) for a in arrays)
        padded = np.full((len(arrays), max_len), np.nan, dtype=float)
        for i, arr in enumerate(arrays):
            padded[i, : len(arr)] = arr
        return np.nanmean(padded, axis=0)

    def plot_all(
        self,
        show: bool = False,
        save_dir: Optional[str] = None,
        filename_suffix: str = "",
        view: Optional[str] = None,
    ) -> None:
        """Plot all configured figures."""
        view = view or self.plot_view
        if view not in self.VALID_VIEWS:
            raise ValueError("plot_view must be 'train-valid', 'epoch', or 'combined'")
        if view in {"train-valid", "combined"}:
            self._plot_train_valid(show=show, save_dir=save_dir, filename_suffix=filename_suffix)
        if view in {"epoch", "combined"}:
            self._plot_epoch(show=show, save_dir=save_dir, filename_suffix=filename_suffix)

    def save_plots(self, epoch: int, save_dir: str, view: Optional[str] = None) -> None:
        """Save current plots for a specific epoch (useful in online mode)."""
        self.plot_all(show=False, save_dir=save_dir, filename_suffix=f"_e{epoch}", view=view)

    def _plot_train_valid(self, show: bool, save_dir: Optional[str], filename_suffix: str) -> None:
        cfg = self.plot_config.get("train-valid", {})
        groups = cfg.get("groups", {})
        for group_name, gcfg in groups.items():
            train_aliases = gcfg.get("train", [])
            valid_aliases = gcfg.get("valid", [])
            metrics = gcfg.get("metrics") or self.default_metrics
            modes = gcfg.get("modes")
            plot_spectrum = bool(gcfg.get("plot_spectrum", False))
            plot_components = bool(gcfg.get("plot_components", False))

            mode_set = set(modes or [])
            if not mode_set:
                for alias in train_aliases + valid_aliases:
                    dest = self.target_config[alias]["destination"]
                    mode_set.update(self.records.get(dest, {}).get(alias, {}).keys())

            for mode in mode_set:
                series_list = []
                series_list.extend(self._collect_series(train_aliases, mode, suffix="train"))
                series_list.extend(self._collect_series(valid_aliases, mode, suffix="valid"))
                if not series_list:
                    continue

                self._plot_metric_subplots(
                    title=f"{group_name} / train-valid / {mode}",
                    metrics=metrics,
                    series_list=series_list,
                    show=show,
                    save_dir=save_dir,
                    filename=f"{group_name}_train_valid_{mode}_metrics{filename_suffix}.png",
                )

                if plot_spectrum:
                    self._plot_latest_spectra(
                        aliases=train_aliases + valid_aliases,
                        mode=mode,
                        show=show,
                        save_dir=save_dir,
                    )
                if plot_components:
                    self._plot_latest_components(
                        aliases=train_aliases + valid_aliases,
                        mode=mode,
                        show=show,
                        save_dir=save_dir,
                    )

    def _plot_epoch(self, show: bool, save_dir: Optional[str], filename_suffix: str) -> None:
        cfg = self.plot_config.get("epoch", {})
        groups = cfg.get("groups", {})
        for group_name, gcfg in groups.items():
            aliases = gcfg.get("aliases", [])
            metrics = gcfg.get("metrics") or self.default_metrics
            modes = gcfg.get("modes")
            plot_spectrum = bool(gcfg.get("plot_spectrum", False))
            plot_components = bool(gcfg.get("plot_components", False))

            mode_set = set(modes or [])
            if not mode_set:
                for alias in aliases:
                    dest = self.target_config[alias]["destination"]
                    mode_set.update(self.records.get(dest, {}).get(alias, {}).keys())

            for mode in mode_set:
                series_list = self._collect_series(aliases, mode, suffix=None)
                if not series_list:
                    continue

                self._plot_metric_subplots(
                    title=f"{group_name} / epoch / {mode}",
                    metrics=metrics,
                    series_list=series_list,
                    show=show,
                    save_dir=save_dir,
                    filename=f"{group_name}_epoch_{mode}_metrics{filename_suffix}.png",
                )

                if plot_spectrum:
                    self._plot_latest_spectra(
                        aliases=aliases,
                        mode=mode,
                        show=show,
                        save_dir=save_dir,
                    )
                if plot_components:
                    self._plot_latest_components(
                        aliases=aliases,
                        mode=mode,
                        show=show,
                        save_dir=save_dir,
                    )

    def _collect_series(self, aliases: Sequence[str], mode: str, suffix: Optional[str]) -> List[Dict[str, Any]]:
        series_list: List[Dict[str, Any]] = []
        for alias in aliases:
            dest = self.target_config[alias]["destination"]
            entries = self.records.get(dest, {}).get(alias, {}).get(mode, [])
            if not entries:
                continue
            label = self.target_config[alias].get("label", alias)
            if suffix:
                label = f"{label}-{suffix}"
            series_list.append({"label": label, "entries": entries})
        return series_list

    def _plot_metric_subplots(
        self,
        title: str,
        metrics: Sequence[str],
        series_list: List[Dict[str, Any]],
        show: bool,
        save_dir: Optional[str],
        filename: Optional[str],
    ) -> None:
        if not metrics:
            return
        n = len(metrics)
        ncols = 2 if n > 1 else 1
        nrows = int(math.ceil(n / ncols))
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7 * ncols, 3.8 * nrows))
        axes = np.atleast_1d(axes).reshape(-1)

        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            for series in series_list:
                entries = series["entries"]
                epochs = [e["epoch"] for e in entries if "epoch" in e]
                values = [float(e.get(metric, np.nan)) for e in entries]
                ax.plot(epochs, values, label=series["label"])
            ax.set_title(metric)
            ax.set_xlabel("epoch")
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend()

        for j in range(len(metrics), len(axes)):
            axes[j].axis("off")

        fig.suptitle(title)
        if save_dir and filename:
            fig.savefig(f"{save_dir}/{filename}", dpi=160, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

    def _plot_latest_spectra(
        self,
        aliases: Sequence[str],
        mode: str,
        show: bool,
        save_dir: Optional[str],
    ) -> None:
        for alias in aliases:
            dest = self.target_config[alias]["destination"]
            entries = self.records.get(dest, {}).get(alias, {}).get(mode, [])
            if not entries:
                continue
            entry = entries[-1]
            epoch = entry.get("epoch", 0)
            if epoch % self.spectrum_interval != 0:
                continue
            if "eigenvalues" not in entry:
                continue
            self._plot_spectrum(entry["eigenvalues"], dest, alias, mode, epoch, save_dir, show)

    def _plot_latest_components(
        self,
        aliases: Sequence[str],
        mode: str,
        show: bool,
        save_dir: Optional[str],
    ) -> None:
        for alias in aliases:
            dest = self.target_config[alias]["destination"]
            entries = self.records.get(dest, {}).get(alias, {}).get(mode, [])
            if not entries:
                continue
            entry = entries[-1]
            epoch = entry.get("epoch", 0)
            if epoch % self.component_interval != 0:
                continue
            components = entry.get("components")
            if components is None:
                continue
            self._plot_components(components, dest, alias, mode, epoch, save_dir, show)

    def _plot_spectrum(
        self,
        eigenvalues: Sequence[float],
        dest: str,
        alias: str,
        mode: str,
        epoch: int,
        save_dir: Optional[str],
        show: bool,
    ) -> None:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(np.arange(1, len(eigenvalues) + 1), eigenvalues, marker="o")
        ax.set_title(f"Spectrum @epoch {epoch} ({dest}/{alias}/{mode})")
        ax.set_xlabel("component")
        ax.set_ylabel("variance")
        ax.grid(True, linestyle="--", alpha=0.4)
        if save_dir:
            fname = f"{dest}_{alias}_{mode}_spectrum_e{epoch}.png"
            fig.savefig(f"{save_dir}/{fname}", dpi=160, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

    def _plot_components(
        self,
        components: np.ndarray,
        dest: str,
        alias: str,
        mode: str,
        epoch: int,
        save_dir: Optional[str],
        show: bool,
    ) -> None:
        fig, ax = plt.subplots(figsize=(6, 4))
        for idx, comp in enumerate(components, start=1):
            ax.plot(comp, label=f"PC{idx}")
        ax.set_title(f"Components @epoch {epoch} ({dest}/{alias}/{mode})")
        ax.set_xlabel("dimension")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.4)
        if save_dir:
            fname = f"{dest}_{alias}_{mode}_components_e{epoch}.png"
            fig.savefig(f"{save_dir}/{fname}", dpi=160, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

