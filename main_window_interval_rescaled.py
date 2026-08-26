"""
Training entry for the proposed interval-target regularization.

Paper objective
---------------

    L = L_main + lambda * L_B

where

    L_B =
        1/N * sum_{s,t,k} (w - f)^2
        +
        1/M * sum_{s,t} |sum_k f - 1|.

The disjoint historical interval width is controlled by
``interval_step``. The main paper setting uses interval_step = 2.
python main_window_interval_rescaled.py \
    --model akt \
    --dataset assist09 \
    --interval_step 2 \
    --window_reg_weight 0.01
"""

import argparse
import importlib
import os
import sys

import train as train_module

from baseline.window_interval_rescaled import (
    AKTIntervalTargets,
    AKTIntervalRescaled,
    FoLiBiIntervalTargets,
    FoLiBiIntervalRescaled,
    SAKTIntervalTargets,
    SAKTIntervalRescaled,
    RobustKTIntervalTargets,
    RobustKTIntervalRescaled,
    SimpleKTIntervalTargets,
    SimpleKTIntervalRescaled,
    SparseKTIntervalTargets,
    SparseKTIntervalRescaled,
    CL4KTIntervalTargets,
    CL4KTIntervalRescaled,
)


def parse_experiment_args():
    """
    Parse only arguments specific to the proposed interval-target method.

    All remaining arguments are passed to the original main.py.
    """

    parser = argparse.ArgumentParser(
        add_help=False
    )

    # ------------------------------------------------------------
    # Delta in:
    #
    #   (0, Delta], (Delta, 2Delta], ...
    #
    # Main paper setting:
    #   Delta = 2
    # ------------------------------------------------------------
    parser.add_argument(
        "--interval_step",
        type=int,
        default=2,
    )

    # ------------------------------------------------------------
    # lambda in:
    #
    #   L = L_main + lambda * L_B
    # ------------------------------------------------------------
    parser.add_argument(
        "--window_reg_weight",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--interval_max_seq_len",
        type=int,
        default=200,
    )

    # ------------------------------------------------------------
    # Embedding dimension for:
    #
    #   interval start
    #   interval end
    #   history length
    # ------------------------------------------------------------
    parser.add_argument(
        "--interval_index_dim",
        type=int,
        default=32,
    )

    # ------------------------------------------------------------
    # Hidden dimension of the target MLP
    # ------------------------------------------------------------
    parser.add_argument(
        "--interval_hidden_dim",
        type=int,
        default=128,
    )

    # Fixed modes in the unified paper equation.
    parser.add_argument(
        "--rescale_mode",
        choices=("explicit", "lb_only"),
        default="explicit",
    )
    parser.add_argument("--rescale_alpha", type=float, default=1.0)
    parser.add_argument("--rescale_beta", type=float, default=0.0)
    parser.add_argument(
        "--run_tag",
        type=str,
        default="",
        help="Optional tag appended to checkpoint/log timestamp for isolated parallel runs.",
    )

    args, remaining = (
        parser.parse_known_args()
    )

    # Keep only the ordinary main.py arguments for the base parser.
    sys.argv = [
        sys.argv[0],
        *remaining,
    ]

    return args


EXP = parse_experiment_args()


# ================================================================
# Load original main
# ================================================================

base_main = importlib.import_module(
    os.environ.get(
        "WINDOW_TARGET_BASE_MAIN",
        "main",
    )
)


if EXP.run_tag:
    _base_datetime = base_main.datetime

    class _TaggedDateTime:
        @staticmethod
        def now(tz=None):
            timestamp = _base_datetime.now(tz)

            class _TaggedTimestamp:
                def strftime(self, fmt):
                    return f"{timestamp.strftime(fmt)}_{EXP.run_tag}"

            return _TaggedTimestamp()

    base_main.datetime = _TaggedDateTime



def common_kwargs():
    """
    Arguments shared by all interval-target backbones.
    """

    return {
        "window_reg_weight":
            EXP.window_reg_weight,

        "interval_step":
            EXP.interval_step,

        "interval_max_seq_len":
            EXP.interval_max_seq_len,

        "interval_index_dim":
            EXP.interval_index_dim,

        "interval_hidden_dim":
            EXP.interval_hidden_dim,
    }


def explicit_rescale_kwargs():
    return {
        "rescale_alpha": EXP.rescale_alpha,
        "rescale_beta": EXP.rescale_beta,
    }

def akt_factory(
    *args,
    **kwargs,
):
    model_class = AKTIntervalRescaled if EXP.rescale_mode == "explicit" else AKTIntervalTargets
    extra_kwargs = explicit_rescale_kwargs() if EXP.rescale_mode == "explicit" else {}
    return model_class(
        *args,
        **common_kwargs(),
        **extra_kwargs,
        **kwargs,
    )


def folibi_factory(
    *args,
    **kwargs,
):
    model_class = FoLiBiIntervalRescaled if EXP.rescale_mode == "explicit" else FoLiBiIntervalTargets
    extra_kwargs = explicit_rescale_kwargs() if EXP.rescale_mode == "explicit" else {}
    return model_class(
        *args,
        **common_kwargs(),
        **extra_kwargs,
        **kwargs,
    )


def sakt_factory(
    *args,
    **kwargs,
):
    model_class = SAKTIntervalRescaled if EXP.rescale_mode == "explicit" else SAKTIntervalTargets
    extra_kwargs = explicit_rescale_kwargs() if EXP.rescale_mode == "explicit" else {}
    return model_class(
        *args,
        **common_kwargs(),
        **extra_kwargs,
        **kwargs,
    )


def robustkt_factory(
    *args,
    **kwargs,
):
    model_class = RobustKTIntervalRescaled if EXP.rescale_mode == "explicit" else RobustKTIntervalTargets
    extra_kwargs = explicit_rescale_kwargs() if EXP.rescale_mode == "explicit" else {}
    return model_class(
        *args,
        **common_kwargs(),
        **extra_kwargs,
        **kwargs,
    )


def simplekt_factory(
    *args,
    **kwargs,
):
    model_class = SimpleKTIntervalRescaled if EXP.rescale_mode == "explicit" else SimpleKTIntervalTargets
    extra_kwargs = explicit_rescale_kwargs() if EXP.rescale_mode == "explicit" else {}
    return model_class(
        *args,
        **common_kwargs(),
        **extra_kwargs,
        **kwargs,
    )


def sparsekt_factory(
    *args,
    **kwargs,
):
    model_class = SparseKTIntervalRescaled if EXP.rescale_mode == "explicit" else SparseKTIntervalTargets
    extra_kwargs = explicit_rescale_kwargs() if EXP.rescale_mode == "explicit" else {}
    return model_class(
        *args,
        **common_kwargs(),
        **extra_kwargs,
        **kwargs,
    )


def cl4kt_factory(
    *args,
    **kwargs,
):
    model_class = CL4KTIntervalRescaled if EXP.rescale_mode == "explicit" else CL4KTIntervalTargets
    extra_kwargs = explicit_rescale_kwargs() if EXP.rescale_mode == "explicit" else {}
    return model_class(
        *args,
        **common_kwargs(),
        **extra_kwargs,
        **kwargs,
    )



base_main.AKT = akt_factory

base_main.folibiKT = (
    folibi_factory
)

base_main.SAKT = (
    sakt_factory
)

base_main.Robustkt = (
    robustkt_factory
)

base_main.simpleKT = (
    simplekt_factory
)

base_main.sparseKT = (
    sparsekt_factory
)

base_main.CL4KT = (
    cl4kt_factory
)




original_model_forward = (
    train_module.model_forward
)


def model_forward_with_interval_aux(
    model,
    batch,
    device,
    model_name,
    num_q=None,
):
    """
    Run the original training forward pass and add the interval
    auxiliary loss when the wrapped model exposes ``window_aux_loss``.

    The wrapped model already stores

        window_aux_loss = lambda * L_B,

    so it must be added only once here.
    """

    result = original_model_forward(
        model,
        batch,
        device,
        model_name,
        num_q=num_q,
    )

    if (
        model.training
        and hasattr(
            model,
            "window_aux_loss",
        )
    ):
        auxiliary_loss = (
            model.window_aux_loss
        )

        if auxiliary_loss is not None:
            (
                loss,
                prediction,
                truth,
                mask,
            ) = result

            loss = (
                loss
                + auxiliary_loss
            )

            return (
                loss,
                prediction,
                truth,
                mask,
            )

    return result


train_module.model_forward = (
    model_forward_with_interval_aux
)



if __name__ == "__main__":
    base_main.main()