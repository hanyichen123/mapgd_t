import time
import json
import numpy as np
from mapgd_tasks import get_mapgd_task_class
from mapgd_predictors import get_mapgd_predictor
from liar_config import EXPERIMENT_CONFIG as LIAR_CONFIG
from jailbreak_config import EXPERIMENT_CONFIG as JAILBREAK_CONFIG
from ethos_config import EXPERIMENT_CONFIG as ETHOS_CONFIG
from gsm8k_config import EXPERIMENT_CONFIG as GSM8K_CONFIG
from aqua_config import EXPERIMENT_CONFIG as AQUA_CONFIG
from svamp_config import EXPERIMENT_CONFIG as SVAMP_CONFIG
from sarcasm_config import EXPERIMENT_CONFIG as SARCASM_CONFIG
from paper_mapgd import PaperAlignedMAPGD


class MAPGDFramework:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("BaselineCompatibleMAPGDFramework is deprecated. Use PaperAlignedMAPGD.")


ChannelAdaptiveAgentWeighting = None
MAPGDUtils = None


class BaselineCompatibleMAPGDFramework(MAPGDFramework):
    def __init__(self, config):
        super().__init__(config)
        if config.get('enable_caaw', True):
            print("✅ Channel-Adaptive Agent Weighting (CAAW) is ENABLED.")
            self.caaw = ChannelAdaptiveAgentWeighting(config)
            self.agent_performance_history = []
        else:
            print("❌ Channel-Adaptive Agent Weighting (CAAW) is DISABLED.")
            self.caaw = None
            self.agent_performance_history = None

    def _build_reproducibility_artifacts(self):
        return {
            'agent_runtime': self.agent_manager.get_reproducibility_manifest(),
            'gradient_coordinator': self.gradient_coordinator.get_reproducibility_manifest(),
            'prompt_expander': self.prompt_expander.get_reproducibility_manifest(),
            'parsing_rules': MAPGDUtils.get_parsing_rules(),
        }

    def optimize_prompt(
        self, initial_prompt, task_name, data_dir, predictor=None, predictor_type='binary_classification'
    ):
        start_time = time.time()

        if 'task_name' not in self.config:
            self.config['task_name'] = task_name

        print(f"🔍 Task name in config: {self.config.get('task_name')}")
        print(f"🔍 Predictor type: {predictor_type}")

        task_class = get_mapgd_task_class(task_name)
        task = task_class(data_dir)
        print(f"🔎 task.data_dir = {task.data_dir}")
        train_examples = task.get_train_examples()
        test_examples = task.get_test_examples()

        print(f"📊 Dataset: {len(train_examples)} train, {len(test_examples)} test examples")

        if predictor is None:
            print(f"🔧 Creating predictor of type: {predictor_type}")
            predictor = get_mapgd_predictor(predictor_type, self.config)
            print(f"✅ Predictor created: {type(predictor)}")

        if predictor is None:
            raise ValueError(f"Failed to create predictor of type: {predictor_type}")

        print(f"🤖 Initializing agents for task: {task_name}")
        self.agent_manager.initialize_agents(initial_prompt)
        reproducibility_artifacts = self._build_reproducibility_artifacts()

        optimization_history = []
        best_overall_prompt = initial_prompt
        best_overall_score = 0.0
        current_best_prompt = initial_prompt
        current_best_score = float("-inf")
        last_iteration_test_score = 0.0

        for iteration in range(self.max_iterations):
            print(f"\n🔄 === Optimization Step {iteration + 1}/{self.max_iterations} ===")
            iteration_start = time.time()

            minibatch_size = min(self.config.get('minibatch_size', 64), len(train_examples))
            minibatch = np.random.choice(train_examples, minibatch_size, replace=False).tolist()
            print(f"📦 Sampled minibatch: {minibatch_size} examples")

            print(f"🔍 Predictor check before gradient generation: {type(predictor)}")
            agent_gradients = self.agent_manager.generate_gradients(minibatch, task, predictor)
            print(f"📈 Generated gradients from {len(agent_gradients)} agents")

            if self.config.get('disable_fusion', False):
                fused_gradients = sum(agent_gradients.values(), [])
                print(f"🚫 Fusion disabled, using {len(fused_gradients)} raw gradients")
            else:
                fused_gradients = self.gradient_coordinator.coordinate_gradients(
                    agent_gradients,
                    caaw_instance=self.caaw,
                    agent_history=self.agent_performance_history,
                    current_iteration=iteration,
                )
                print(f"🔀 Fused to {len(fused_gradients)} coordinated gradients")

            print(f"🔀 Fused to {len(fused_gradients)} coordinated gradients")

            candidates = self.prompt_expander.expand(current_prompt=current_best_prompt, gradients=fused_gradients)
            print(f"🌱 Generated {len(candidates)} candidate prompts")

            eval_sample_size = min(32, len(train_examples))
            eval_samples = np.random.choice(train_examples, eval_sample_size, replace=False).tolist()

            scores = self.candidate_selector.evaluate_prompts(candidates, eval_samples, task, predictor)
            print(f"📊 Evaluated {len(candidates)} candidates")

            prompt_score_pairs = list(zip(scores, candidates))
            prompt_score_pairs.sort(reverse=True)

            candidate_best_prompt = prompt_score_pairs[0][1]
            candidate_best_score = prompt_score_pairs[0][0]

            previous_prompt = current_best_prompt
            previous_score = current_best_score
            score_margin = self.config.get('prompt_update_min_margin', 0.0)
            if candidate_best_score >= previous_score + score_margin:
                current_best_prompt = candidate_best_prompt
                current_best_score = candidate_best_score
            else:
                current_best_prompt = previous_prompt
                current_best_score = previous_score
                print(
                    '[Selector] Keeping current prompt '
                    f"(candidate={candidate_best_score:.4f}, current={previous_score:.4f})"
                )

            print(f"📈 Selected best prompt with score: {current_best_score:.4f}")

            default_test_eval = self.config.get('test_eval_size', 150)
            test_eval_size = min(default_test_eval, len(test_examples))
            test_samples = test_examples[:test_eval_size]
            test_score, _, _, _ = task.evaluate_prompt(current_best_prompt, test_samples, predictor, n=test_eval_size)
            print(f"📊 Best prompt test score: {test_score:.4f}")
            if self.caaw and self.agent_performance_history is not None:
                performance_gain = test_score - last_iteration_test_score
                print(f"[CAAW] Performance gain this iteration: {performance_gain:.4f}")
                fused_agents = getattr(self.caaw, "last_fusion_agents", set()) or set()
                if not fused_agents:
                    fused_agents = {f'agent_{i}' for i in range(len(self.agent_manager.agents))}
                weight_lookup = {}
                for record in getattr(self.caaw, "last_fusion_records", []):
                    weight_lookup.update(record.get('agent_weights', {}))
                for agent_id in fused_agents:
                    self.agent_performance_history.append(
                        {
                            'iteration': iteration,
                            'agent_id': agent_id,
                            'improvement': performance_gain,
                            'test_score': test_score,
                            'accepted': True,
                            'weight_used': weight_lookup.get(agent_id),
                        }
                    )
                last_iteration_test_score = test_score
            if test_score > best_overall_score:
                best_overall_prompt = current_best_prompt
                best_overall_score = test_score

            iteration_time = time.time() - iteration_start

            iteration_result = {
                'iteration': iteration + 1,
                'best_prompt': current_best_prompt,
                'train_score': current_best_score,
                'test_score': test_score,
                'num_candidates': len(candidates),
                'minibatch_size': minibatch_size,
                'eval_sample_size': eval_sample_size,
                'iteration_time': iteration_time,
            }

            optimization_history.append(iteration_result)

            print(f"✅ Step {iteration + 1} Results:")
            print(f"   📈 Train Score: {current_best_score:.4f}")
            print(f"   🎯 Test Score: {test_score:.4f}")
            print(f"   ⏱️  Time: {iteration_time:.2f}s")
            print(f"   📝 Best Prompt: {current_best_prompt}")

            if self.convergence_monitor.check_convergence(test_score):
                print(f"🎯 Converged after {iteration + 1} optimization steps")
                break

        total_time = time.time() - start_time

        final_result = {
            'framework': 'MAPGD-Baseline-Compatible',
            'best_prompt': best_overall_prompt,
            'final_test_score': best_overall_score,
            'optimization_history': optimization_history,
            'total_iterations': len(optimization_history),
            'total_time': total_time,
            'config': self.config,
            'reproducibility': reproducibility_artifacts,
            'baseline_compliant': True,
            'multi_agent_fusion': True,
        }

        print('\n🏁 MAPGD Optimization Complete!')
        print('📊 Final Results:')
        print(f"   🏆 Best Test Score: {best_overall_score:.4f}")
        print(f"   🔢 Total Steps: {len(optimization_history)}")
        print(f"   ⏱️  Total Time: {total_time:.2f}s")
        print(f"   📝 Final Prompt: {best_overall_prompt}")

        return final_result


def run_baseline_experiment(task="jailbreak"):
    if task == "liar":
        config = LIAR_CONFIG.copy()
        initial_prompt = LIAR_CONFIG['initial_prompt']
        predictor_type = 'binary_classification'

    elif task == "jailbreak":
        config = JAILBREAK_CONFIG.copy()
        initial_prompt = JAILBREAK_CONFIG['initial_prompt']
        predictor_type = 'jailbreak_classification'

    elif task == "ethos":
        config = ETHOS_CONFIG.copy()
        initial_prompt = ETHOS_CONFIG['initial_prompt']
        predictor_type = 'binary_classification'

    elif task == "sarcasm":
        config = SARCASM_CONFIG.copy()
        initial_prompt = SARCASM_CONFIG['initial_prompt']
        predictor_type = 'binary_classification'

    elif task == "gsm8k":
        config = GSM8K_CONFIG.copy()
        initial_prompt = GSM8K_CONFIG['initial_prompt']
        predictor_type = 'math_reasoning'
    elif task == "svamp":
        config = SVAMP_CONFIG.copy()
        initial_prompt = SVAMP_CONFIG['initial_prompt']
        predictor_type = 'math_reasoning'
    elif task == "aqua":
        config = AQUA_CONFIG.copy()
        initial_prompt = AQUA_CONFIG['initial_prompt']
        predictor_type = 'aqua_reasoning'

    else:
        raise ValueError(f"Unsupported task: {task}")

    print("🚀 Starting MAPGD Experiment with Baseline-Compatible Settings")
    print("=" * 60)
    print("📊 Experiment Configuration (Baseline Match):")
    print(f"  • Optimization Steps: {config['max_iterations']}")
    print(f"  • Beam Size: {config['beam_size']}")
    print(f"  • Minibatch Size: {config['minibatch_size']}")
    print(f"  • Error Group Size: {config['error_group_size']}")
    print(f"  • Gradients per Group: {config['gradients_per_group']}")
    print(f"  • Monte Carlo Samples: {config['monte_carlo_samples']}")
    print(f"  • Successor Candidates: {config['successor_candidates']}")
    print(f"  • Selection Strategy: {config['selection_strategy']}")
    print("=" * 60)

    try:
        predictor = get_mapgd_predictor(predictor_type, config)
        task_class = get_mapgd_task_class(config["task_name"])
        task_processor = task_class(config["data_dir"])
        train_examples = task_processor.get_train_examples()
        test_examples = task_processor.get_test_examples()

        print("\n🎯 Starting paper-aligned optimization...")
        print(f"Initial prompt: {initial_prompt[:80]}...")
        framework = PaperAlignedMAPGD(
            config=config,
            task=task_processor,
            predictor=predictor,
        )
        results = framework.optimize(
            initial_prompt=initial_prompt,
            train_examples=train_examples,
            test_examples=test_examples,
        )

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        result_file = f"baseline_{task}_results_{timestamp}.json"

        with open(result_file, 'w', encoding='utf-8') as f:
            results_to_save = {
                'experiment_type': 'paper_aligned_mapgd',
                'config': config,
                'results': results,
                'timestamp': timestamp,
            }
            json.dump(results_to_save, f, indent=2, ensure_ascii=False)

        print(f"\n💾 Results saved to: {result_file}")
        return results

    except Exception as e:
        print(f"✗ Baseline experiment failed: {e}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        default="jailbreak",
        choices=["liar", "jailbreak", "ethos", "sarcasm", "gsm8k", "aqua", "svamp"],
    )
    args = parser.parse_args()

    print("🔬 MAPGD Paper-Aligned Experiment")
    print(f"🎯 Task: {args.task}")
    print()

    results = run_baseline_experiment(task=args.task)
    if results:
        print("\n✅ Paper-aligned experiment completed successfully!")
        print(f"🎯 Final test score: {results['final_test_score']:.4f}")
    else:
        print("\n❌ Baseline-compatible experiment failed!")
