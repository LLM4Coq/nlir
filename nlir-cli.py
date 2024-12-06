import hydra
import weave
import sys
import json
import numpy as np
from hydra.core.hydra_config import HydraConfig
from pytanque import Pytanque, PetanqueError
from nlir.agent import Ghost, GPT
from nlir.petanque import TacticEnv, TemplateEnv
from nlir.search import naive_search, beam_search
from pathlib import Path
from datetime import datetime
from omegaconf import DictConfig
from functools import partial


def check_benchmark(
    pet: Pytanque, wk_path: Path, cfg: DictConfig
) -> list[tuple[Path, str]]:
    errors = []
    theorems = []
    counter = 0
    for thms in cfg.benchmark:
        file_path = Path(wk_path, thms.file).absolute()
        for thm in thms.theorems:
            if counter in range(cfg.start_theorem,cfg.end_theorem):
                try:
                    pet.start(str(file_path), thm)
                    theorems.append((file_path, thm))
                except PetanqueError as err:
                    errors.append(f"- File {thms.file} {thm}: {err.message}")
            counter += 1
    if not errors:
        print(f"Benchmarking {len(theorems)} theorems starting from {cfg.start_theorem} in {len(cfg.benchmark)} files")
    else:
        print(f"Config contains the following errors:", file=sys.stderr)
        print("\n".join(errors), file=sys.stderr)
        sys.exit(1)
    return theorems


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    wk_path = Path(cfg.workspace).expanduser().absolute()
    pet = Pytanque(cfg.petanque.address, cfg.petanque.port)
    pet.connect()
    pet.set_workspace(False, str(wk_path))

    match cfg.search.kind:
        case "tactics":
            env_cls = TacticEnv
        case "template":
            env_cls = TemplateEnv
        case _:
            raise RuntimeError(
                "search.kind config should be one of [tactics, template]"
            )

    match cfg.search.mode:
        case "naive":
            search = naive_search
        case "beam":
            search = partial(
                beam_search,
                beam_size=cfg.search.beam_size,
                n_reponses=cfg.search.n_responses,
            )
        case _:
            raise RuntimeError("search.mode config should be one of [naive, beam]")

    if cfg.theorem and cfg.file:
        # Only prove thm from file (ignore benchmark)
        dt = datetime.now().strftime("%y%m%d-%H%M%S")
        log_path = Path(cfg.log_dir, f"{cfg.file}:{cfg.theorem}_{dt}.jsonl").absolute()
        file_path = Path(wk_path, cfg.file)
        if cfg.replay:
            source_path = Path(cfg.replay)
            agent = Ghost(source_path.resolve())
        else:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            agent = GPT(
                str(log_path),
                cfg.agent,
            )

        env = env_cls(
            pet, str(wk_path), str(file_path), cfg.theorem, cfg.petanque.context
        )

        print(f"Try to prove {cfg.theorem} from {cfg.file}")
        status = search(agent, env, cfg.search.max_steps)
        print(f"\n\n--- Success: {status.success} ---")
        print(f"Proof: {status.proof}")
        print("---\n\n")
        sys.exit(0)

    elif cfg.benchmark:
        # Try the full benchmark
        if cfg.weave:
            model_id = cfg.agent.model_id
            name_expe = f"{model_id.split('/')[-1]}:{cfg.benchmark[0].file}"
            weave.init(name_expe)
        log_dir = HydraConfig.get().runtime.output_dir

        results = {"names": [], "success": [], "steps": []}
        theorems = check_benchmark(pet, wk_path, cfg)
        res_path = Path(log_dir, f"eval_results_{cfg.start_theorem}_{len(theorems)}.json")

        for file_path, thm in theorems:
            print(f"\n\nTrying to prove {thm} from {file_path.stem}")
            env = env_cls(pet, str(wk_path), str(file_path), thm, cfg.petanque.context)
            log_path = Path(log_dir, f"{file_path.stem}:{thm}.jsonl").absolute()
            agent = GPT(
                str(log_path),
                cfg.agent,
            )
            with weave.attributes({"file": file_path.stem, "thm": thm}):
                status = search(agent, env, cfg.search.max_steps)
            results["names"].append(f"{env.file}:{env.thm}")
            results["success"].append(status.success)
            results["steps"].append(status.steps)
            with open(res_path, "w") as rf:
                json.dump(results, rf, indent=2)

        print(f"\n\n--- Summary ---")
        print(f"Theorems: {len(theorems)}")
        print(f"Successes: {np.sum(results['success'])}")
        print(f"Average number of steps: {np.mean(results['steps'])}")
        print("---\n\n")
        sys.exit(0)

    else:
        print("Nothing to do. Try --help for more.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
