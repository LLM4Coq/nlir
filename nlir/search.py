from .petanque import Env, TemplateEnv
from .agent import LLM
from dataclasses import dataclass
from typing import List
import re
from typing import Iterable
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionMessage
from .prompts import comparison_prompts
from functools import partial
from ast import literal_eval
import weave

@dataclass
class Status:
    steps: int
    success: bool
    proof: str

@weave.op()
def naive_search(agent: LLM, env: Env, max_steps: int) -> Status:
    response = agent.response(env.prompt)  # Initial step
    for step in range(max_steps):
        env.exec(response)
        proof = " ".join(env.proof)
        print(f"\nNaive search, step {step}:\n{proof}")
        if env.proof_finished:
            if env.check_proof():
                agent.log({"role": "user", "content": f"Final Proof: {proof}"})
                return Status(step, True, proof)
            else:
                agent.log({"role": "user", "content": f"Failed Proof: {proof}"})
                return Status(step, False, proof)
        else:
            if len(str(env.prompt)) > 100000:
                # prompt is too big!
                break
            response = agent.response(env.prompt)

    if not env.proof_finished:
        proof = " ".join(env.proof)
        agent.log({"role": "user", "content": f"Partial Proof: {proof}"})
        return Status(max_steps, False, proof)

    raise RuntimeError("Unreachable code.")


def create_comparison_prompt(
    list_env: list[Env],
) -> Iterable[ChatCompletionMessageParam]:
    """
    Build the comparison prompt from the list of environments.
    """
    system_prompt = comparison_prompts.comparison_system_prompt
    intro = comparison_prompts.user_prompt.format(
        theorem_code=list_env[0].thm_code,
    )
    attempts = "\n\n".join(
        [
            f"Here is the {i}th attempt:\n\n{env.prompt_for_comparison}"
            for i, env in enumerate(list_env)
        ]
    )
    content = "\n\n".join([intro, attempts])
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]


def parse_comparison(
    message: ChatCompletionMessage) -> List[int]:
    if message.content:
        match = re.match(
            r".*(\[\s*(?:\d+\s*(?:,\s*\d+\s*)*)?\]).*", message.content, re.DOTALL
        )
        if match:
            try:
                return literal_eval(match[1])
            except ValueError:
                pass
        # model didn't format the list properly, try alternative way of parsing
        # simply get all numbers followed by a comma, end of string or newline
        # TBC
        if "Response" in message.content:
            to_parse = message.content.split("Response")[-1]
        else:
            to_parse = message.content
        match = re.findall(r"([0-9]+)(,|$|\n)", to_parse, re.DOTALL)
        parsed = [int(el[0]) for el in match]
        return parsed  
    return []

@weave.op()
def sort_LLM(new_beam: list[Env], agent: LLM) -> list[Env]:
    comparison_prompt = create_comparison_prompt(new_beam)
    response = agent.response(comparison_prompt)
    perm_indices = parse_comparison(response)
    beam_size = len(new_beam)
    if len(perm_indices) < beam_size or not all([i < beam_size for i in perm_indices]):
        perm_indices = list(range(beam_size))
    return [new_beam[i] for i in perm_indices]


def sort_holes(new_beam: list[TemplateEnv]) -> list[TemplateEnv]:
    return sorted(new_beam, key=lambda x: len(x.holes))

@weave.op()
def expand_beam(
    beam: list[Env], agent: LLM, n_reponses: int
) -> list[Env]:
    new_beam = []
    for env in beam:
        responses = agent.multi_responses(env.prompt, n_reponses)
        for response in responses:
            env_copy = env.deepcopy()
            env_copy.exec(response)
            proof = " ".join(env_copy.proof)
            print(proof)
            if env_copy.proof_finished:
                if env_copy.check_proof():
                    agent.log({"role": "user", "content": f"Final Proof: {proof}"})
                    return [env_copy]
            else:
                try:
                    if len(str(env_copy.prompt)) > 100000:
                        # prompt is too big, env will not be added to the beam
                        continue
                    elif env_copy.proof in [e.proof for e in new_beam]:
                        # avoid adding duplicate proofs
                        continue
                except RuntimeError:
                    continue 
            new_beam.append(env_copy)
    return new_beam

def bs_name(call):
    return f"BS-{call.attributes['thm']}-{call.attributes['file']}"

@weave.op(call_display_name=bs_name)
def beam_search(
    agent: LLM,
    env: Env,
    max_steps: int,
    beam_size: int,
    n_reponses: int,
    sorting_holes=False,
) -> Status:
    sort_beam = partial(sort_LLM, agent=agent)
    if sorting_holes:
        sort_beam = sort_holes
    beam = [env]
    for step in range(max_steps):
        print(f"\nBeam search, step {step}:\n")
        # expand bean
        new_beam = expand_beam(beam, agent, n_reponses)
        if new_beam:
            if len(new_beam) <= beam_size:
                if new_beam[0].proof_finished:
                    proof = " ".join(new_beam[0].proof)
                    return Status(step, True, proof)

            else: 
                # sort new_beam
                beam = sort_beam(new_beam)
                beam = beam[:beam_size]
                print("sorted beam:")
                for b in beam:
                    print(" ".join(b.proof))
        else:
            proof = " ".join(beam[0].proof)
            agent.log(
                {"role": "user", "content": f"Failed Proof (empty beam): {proof}"}
            )
            return Status(step, False, proof)

    #beam = sort_holes(beam)
    proof = " ".join(beam[0].proof)
    agent.log({"role": "user", "content": f"Partial Proof: {proof}"})
    return Status(max_steps, False, proof)
