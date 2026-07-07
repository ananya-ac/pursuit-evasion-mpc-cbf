import argparse
import contextlib
import csv
import io
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

from simulation import Simulation


DEFAULT_PURSUER_COUNTS = [2, 4, 6, 8, 10]
DEFAULT_EVADER_VELOCITIES = [1, 2, 3, 4, 5]
HEATMAP_FIGSIZE = (8, 6)
AXIS_LABEL_FONTSIZE = 18
TICK_FONTSIZE = 15
ANNOTATION_FONTSIZE = 15
COLORBAR_FONTSIZE = 16


def determine_outcome(simulation):
    if simulation.solver_failed:
        return "solver_failure"
    if simulation.collided:
        return "lost_collision"
    if simulation.captured:
        return "won_capture"
    if simulation.escaped:
        return "lost_escape"
    if simulation.touchdown:
        return "lost_touchdown"
    return "incomplete"


def parse_count_list(raw_value):
    if raw_value is None:
        return DEFAULT_PURSUER_COUNTS
    values = [int(part.strip()) for part in raw_value.split(",") if part.strip()]
    if not values:
        raise ValueError("Pursuer count list cannot be empty.")
    return values


def parse_velocity_list(raw_value):
    if raw_value is None:
        return DEFAULT_EVADER_VELOCITIES
    values = [int(part.strip()) for part in raw_value.split(",") if part.strip()]
    if not values:
        raise ValueError("Evader velocity list cannot be empty.")
    return values


def run_single_game(
    game_idx,
    base_seed,
    config,
    num_pursuers,
    evader_v_max,
    verbose_games,
    return_simulation=False,
):
    seed = base_seed + game_idx
    simulation = Simulation(
        config=config,
        solver_mode="pursuit_evasion",
        seed=seed,
        num_pursuers=int(num_pursuers),
        pursuer_v_max=2.0,
        pursuer_a_max=1.0,
        evader_v_max=float(evader_v_max),
        evader_a_max=1.0,
    )

    if verbose_games:
        simulation.run()
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            simulation.run()

    outcome = determine_outcome(simulation)
    steps_completed = max(len(simulation.history_evader) - 1, 0)

    row = {
        "game_idx": game_idx,
        "seed": seed,
        "config": config,
        "num_pursuers": num_pursuers,
        "pursuer_v_max": 2.0,
        "pursuer_a_max": 1.0,
        "evader_v_max": evader_v_max,
        "evader_a_max": 1.0,
        "outcome": outcome,
        "steps_completed": steps_completed,
        "captured": int(simulation.captured),
        "capture_step": simulation.capture_step,
        "escaped": int(simulation.escaped),
        "escape_step": simulation.escape_step,
        "collided": int(simulation.collided),
        "collision_step": simulation.collision_step,
        "collision_kind": simulation.collision_kind,
        "collision_agents": simulation.collision_agents,
        "solver_failed": int(simulation.solver_failed),
        "solver_failure_step": simulation.solver_failure_step,
        "final_evader_x": float(simulation.evader_state[0]),
        "final_evader_y": float(simulation.evader_state[1]),
    }
    if return_simulation:
        return row, simulation
    return row


def build_default_paths(config, n_games):
    stem = f"pursuer_count_grid_config{config}_{n_games}games"
    return (
        f"{stem}.csv",
        f"{stem}_capture_count_heatmap.png",
        f"{stem}_capture_rate_heatmap.png",
    )


def build_default_gif_dir(config, n_games):
    return os.path.join("gifs", f"pursuer_count_grid_config{config}_{n_games}games")


def save_rows_csv(rows, output_csv):
    fieldnames = [
        "game_idx",
        "seed",
        "config",
        "num_pursuers",
        "pursuer_v_max",
        "pursuer_a_max",
        "evader_v_max",
        "evader_a_max",
        "outcome",
        "steps_completed",
        "captured",
        "capture_step",
        "escaped",
        "escape_step",
        "collided",
        "collision_step",
        "collision_kind",
        "collision_agents",
        "solver_failed",
        "solver_failure_step",
        "final_evader_x",
        "final_evader_y",
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_capture_metrics(rows, pursuer_counts, evader_velocities, n_games):
    capture_counts = np.zeros((len(pursuer_counts), len(evader_velocities)), dtype=int)
    capture_rates = np.zeros((len(pursuer_counts), len(evader_velocities)), dtype=float)

    for i, count in enumerate(pursuer_counts):
        for j, e_vel in enumerate(evader_velocities):
            captures = sum(
                1
                for row in rows
                if row["num_pursuers"] == count
                and row["evader_v_max"] == e_vel
                and row["outcome"] == "won_capture"
            )
            capture_counts[i, j] = captures
            capture_rates[i, j] = captures / float(n_games)

    return capture_counts, capture_rates


def save_heatmap(matrix, pursuer_counts, evader_velocities, title, colorbar_label, fmt, output_png):
    fig, ax = plt.subplots(figsize=HEATMAP_FIGSIZE)
    image = ax.imshow(matrix, cmap="Reds", origin="upper")

    ax.set_xticks(np.arange(len(evader_velocities)))
    ax.set_yticks(np.arange(len(pursuer_counts)))
    ax.set_xticklabels(evader_velocities)
    ax.set_yticklabels(pursuer_counts)
    ax.set_xlabel("Evader Max Velocity", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Number of Pursuers", fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            ax.text(
                j,
                i,
                format(value, fmt),
                ha="center",
                va="center",
                color="black",
                fontsize=ANNOTATION_FONTSIZE,
            )

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(colorbar_label, fontsize=COLORBAR_FONTSIZE)
    colorbar.ax.tick_params(labelsize=TICK_FONTSIZE)
    fig.tight_layout()
    fig.savefig(output_png, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Run a pursuit-evasion sweep over number of pursuers and save capture heatmaps."
    )
    parser.add_argument("--n-games", type=int, default=3, help="Games per pursuer count.")
    parser.add_argument(
        "--config",
        type=int,
        default=3,
        help="Simulation config to use. Default is 3 because config 2 only supports 4 pursuers.",
    )
    parser.add_argument(
        "--pursuer-counts",
        type=str,
        default=None,
        help="Comma-separated pursuer counts. Default: 4,6,8,10",
    )
    parser.add_argument(
        "--evader-velocities",
        type=str,
        default=None,
        help="Comma-separated evader max velocities. Default: 1,2,3,4,5",
    )
    parser.add_argument("--seed", type=int, default=0, help="Base seed for the sweep.")
    parser.add_argument("--output-csv", type=str, default=None, help="Per-game CSV output path.")
    parser.add_argument(
        "--output-count-heatmap",
        type=str,
        default=None,
        help="Capture-count heatmap output path.",
    )
    parser.add_argument(
        "--output-rate-heatmap",
        type=str,
        default=None,
        help="Capture-rate heatmap output path.",
    )
    parser.add_argument(
        "--take-gifs",
        action="store_true",
        help="Save one representative GIF and one solver-failure debug GIF when encountered.",
    )
    parser.add_argument(
        "--gif-dir",
        type=str,
        default=None,
        help="Directory to store GIFs. Default: gifs/pursuer_count_grid_configX_Ygames",
    )
    parser.add_argument(
        "--verbose-games",
        action="store_true",
        help="Print each game's internal simulation logs.",
    )
    args = parser.parse_args()

    pursuer_counts = parse_count_list(args.pursuer_counts)
    evader_velocities = parse_velocity_list(args.evader_velocities)
    default_csv, default_count_heatmap, default_rate_heatmap = build_default_paths(
        args.config, args.n_games
    )
    output_csv = args.output_csv or default_csv
    output_count_heatmap = args.output_count_heatmap or default_count_heatmap
    output_rate_heatmap = args.output_rate_heatmap or default_rate_heatmap
    gif_dir = args.gif_dir or build_default_gif_dir(args.config, args.n_games)
    if args.take_gifs:
        os.makedirs(gif_dir, exist_ok=True)

    rows = []
    seed_offset = 0
    saved_solver_failure_gif = False
    saw_solver_failure = False
    for count in pursuer_counts:
        for e_vel in evader_velocities:
            for game_idx in range(args.n_games):
                need_representative_gif = args.take_gifs and game_idx == 0
                need_solver_failure_debug_gif = args.take_gifs and not saved_solver_failure_gif
                need_simulation = need_representative_gif or need_solver_failure_debug_gif
                result = run_single_game(
                    game_idx=game_idx,
                    base_seed=args.seed + seed_offset,
                    config=args.config,
                    num_pursuers=count,
                    evader_v_max=e_vel,
                    verbose_games=args.verbose_games,
                    return_simulation=need_simulation,
                )
                if need_simulation:
                    row, simulation = result
                    if need_representative_gif:
                        gif_name = f"pursuers_{count}_evader_v{e_vel}.gif"
                        gif_path = os.path.join(gif_dir, gif_name)
                        with contextlib.redirect_stdout(io.StringIO()):
                            simulation.generate_animation(video_filename=gif_path)
                    if (
                        need_solver_failure_debug_gif
                        and row["outcome"] == "solver_failure"
                    ):
                        saw_solver_failure = True
                        if len(simulation.history_evader) > 1:
                            failure_gif_name = (
                                f"solver_failure_pursuers_{count}_evader_v{e_vel}_game{game_idx}.gif"
                            )
                            failure_gif_path = os.path.join(gif_dir, failure_gif_name)
                            with contextlib.redirect_stdout(io.StringIO()):
                                simulation.generate_animation(video_filename=failure_gif_path)
                            saved_solver_failure_gif = True
                else:
                    row = result
                    if row["outcome"] == "solver_failure":
                        saw_solver_failure = True
                rows.append(row)
            seed_offset += args.n_games

    save_rows_csv(rows, output_csv)
    capture_counts, capture_rates = build_capture_metrics(
        rows, pursuer_counts, evader_velocities, args.n_games
    )
    save_heatmap(
        capture_counts,
        pursuer_counts,
        evader_velocities,
        title=f"Capture Count by Pursuer Count and Evader Velocity ({args.n_games} Games Each)",
        colorbar_label="Number of Victories (Capture)",
        fmt="d",
        output_png=output_count_heatmap,
    )
    save_heatmap(
        capture_rates,
        pursuer_counts,
        evader_velocities,
        title=f"Capture Rate by Pursuer Count and Evader Velocity ({args.n_games} Games Each)",
        colorbar_label="Capture Rate",
        fmt=".2f",
        output_png=output_rate_heatmap,
    )

    print(f"Completed pursuer-count sweep for config {args.config}.")
    print(f"CSV results written to: {output_csv}")
    print(f"Capture-count heatmap written to: {output_count_heatmap}")
    print(f"Capture-rate heatmap written to: {output_rate_heatmap}")
    if args.take_gifs:
        print(f"Representative GIFs written to: {gif_dir}")
        if saved_solver_failure_gif:
            print("Also wrote one solver-failure debug GIF.")
        elif saw_solver_failure:
            print(
                "Solver-failure games were encountered, but none had enough trajectory history "
                "to render a debug GIF."
            )
        else:
            print("No solver-failure game was encountered, so no solver-failure debug GIF was written.")
    print("Outcome counts:")
    for outcome, count in sorted(Counter(row["outcome"] for row in rows).items()):
        print(f"{outcome}: {count}")


if __name__ == "__main__":
    main()
