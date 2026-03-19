import argparse
import re

import pandas as pd
from vllm import LLM, SamplingParams

from green_score.utils import clean_responses, make_prompt


CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['from'] == 'human' %}\n{{ '<|user|>\n' + message['value'] + eos_token }}\n{% elif message['from'] == 'system' %}\n{{ '<|system|>\n' + message['value'] + eos_token }}\n{% elif message['from'] == 'gpt' %}\n{{ '<|assistant|>\n'  + message['value'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"


class GREENVLLM:
    def __init__(self, model_name):
        self.model_name = model_name
        self.max_length = 2048
        self.categories = [
            "Clinically Significant Errors",
            "Clinically Insignificant Errors",
            "Matched Findings",
        ]
        self.sub_categories = [
            "(a) False report of a finding in the candidate",
            "(b) Missing a finding present in the reference",
            "(c) Misidentification of a finding's anatomic location/position",
            "(d) Misassessment of the severity of a finding",
            "(e) Mentioning a comparison that isn't in the reference",
            "(f) Omitting a comparison detailing a change from a prior study",
        ]
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=1,
            trust_remote_code=False if "Phi" in model_name else True,
            max_model_len=self.max_length,
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.tokenizer.chat_template = CHAT_TEMPLATE
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.clean_up_tokenization_spaces = True
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=self.max_length,
        )

    def build_prompts(self, refs, hyps):
        prompts = [make_prompt(ref, hyp) for ref, hyp in zip(refs, hyps)]
        return [
            self.tokenizer.apply_chat_template(
                [{"from": "human", "value": prompt}, {"from": "gpt", "value": ""}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in prompts
        ]

    def generate(self, refs, hyps):
        prompts = self.build_prompts(refs, hyps)
        outputs = self.llm.generate(prompts, self.sampling_params)
        return [clean_responses(output.outputs[0].text) for output in outputs]

    def compute_error_count(self, response):
        _, sig_errors = self.parse_error_counts(response, self.categories[0])
        matched_findings, _ = self.parse_error_counts(response, self.categories[2])
        return sig_errors + [matched_findings]

    def compute_green(self, response):
        sig_present, sig_errors = self.parse_error_counts(response, self.categories[0])
        matched_findings, _ = self.parse_error_counts(response, self.categories[2])

        if matched_findings == 0:
            return 0

        if sig_present is None or matched_findings is None:
            return None

        return matched_findings / (matched_findings + sum(sig_errors))

    def parse_error_counts(self, text, category, for_reward=False):
        if category not in self.categories:
            raise ValueError(
                f"Category {category} is not a valid category. Please choose from {self.categories}."
            )

        pattern = rf"\[{category}\]:\s*(.*?)(?:\n\s*\n|\Z)"
        category_text = re.search(pattern, text, re.DOTALL)

        sum_counts = 0
        sub_counts = [0 for _ in range(6)]

        if not category_text:
            if for_reward:
                return None, None
            return sum_counts, sub_counts
        if category_text.group(1).startswith("No"):
            return sum_counts, sub_counts

        if category == "Matched Findings":
            counts = re.findall(r"^\b\d+\b(?=\.)", category_text.group(1))
            if len(counts) > 0:
                sum_counts = int(counts[0])
            return sum_counts, sub_counts
        else:
            sub_categories = [s.split(" ", 1)[0] + " " for s in self.sub_categories]
            matches = sorted(re.findall(r"\([a-f]\) .*", category_text.group(1)))

            if len(matches) == 0:
                matches = sorted(re.findall(r"\([1-6]\) .*", category_text.group(1)))
                sub_categories = [
                    f"({i})" + " " for i in range(1, len(self.sub_categories) + 1)
                ]

            for position, sub_category in enumerate(sub_categories):
                for match in range(len(matches)):
                    if matches[match].startswith(sub_category):
                        count = re.findall(r"(?<=: )\b\d+\b(?=\.)", matches[match])
                        if len(count) > 0:
                            sub_counts[position] = int(count[0])
            return sum(sub_counts), sub_counts

    def score_dataframe(self, input_df):
        completions = self.generate(
            refs=input_df["reference"].tolist(),
            hyps=input_df["hypothesis"].tolist(),
        )
        green_scores = [self.compute_green(response) for response in completions]
        error_counts = pd.DataFrame(
            [self.compute_error_count(response) for response in completions],
            columns=self.sub_categories + ["Matched Findings"],
        )
        return pd.DataFrame(
            {
                "example_id": input_df["example_id"].tolist(),
                "green_analysis": completions,
                "green_score": green_scores,
                **error_counts,
            }
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Run GREEN scoring with vLLM.")
    parser.add_argument("--input_file", required=True, help="Path to input CSV.")
    parser.add_argument("--output_file", required=True, help="Path to output CSV.")
    parser.add_argument(
        "--model_name",
        default="StanfordAIMI/GREEN-RadLlama2-7b",
        help="Hugging Face model name or local checkpoint path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_df = pd.read_csv(args.input_file)
    if input_df.empty:
        pd.DataFrame(columns=["example_id", "green_analysis", "green_score"]).to_csv(
            args.output_file, index=False
        )
        return
    scorer = GREENVLLM(model_name=args.model_name)
    result_df = scorer.score_dataframe(input_df)
    result_df.to_csv(args.output_file, index=False)


if __name__ == "__main__":
    main()
