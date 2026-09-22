def minimal_gsm8k_test():
    print("🧪 Minimal GSM8k Test")
    print("=" * 30)

    print("\n1️⃣ Testing Math Predictor...")
    try:
        from gsm8k_config import EXPERIMENT_CONFIG
        from mapgd_predictors import get_mapgd_predictor

        predictor = get_mapgd_predictor('math_reasoning', EXPERIMENT_CONFIG)
        print(f"✅ Predictor created: {type(predictor)}")

        test_example = {
            'text': 'If John has 5 apples and gives away 2, how many does he have left?',
            'numerical_answer': 3,
        }

        test_prompt = """
        Solve this math problem step by step.
        Problem: {text}
        Show your work and end with: #### [answer]
        """

        result = predictor.inference(test_example, test_prompt)
        print('✅ Predictor inference works')
        print(f"   Result: {result[:100]}...")

    except Exception as e:
        print(f"❌ Math predictor test failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    print("\n2️⃣ Testing GSM8k Task Evaluation...")
    try:
        from mapgd_tasks import MAPGDGsm8kTask

        class MockGsm8kTask(MAPGDGsm8kTask):
            def get_train_examples(self):
                return [
                    {
                        'id': 'test-1',
                        'text': 'John has 5 apples and gives away 2. How many apples does he have left?',
                        'answer': 'John has 5 - 2 = 3 apples left. #### 3',
                        'numerical_answer': 3.0,
                    }
                ]

            def get_test_examples(self):
                return self.get_train_examples()

        task = MockGsm8kTask()
        train_examples = task.get_train_examples()
        print(f"✅ Mock task created with {len(train_examples)} examples")

        score, texts, true_answers, pred_answers = task.evaluate_prompt(test_prompt, train_examples, predictor, n=1)
        print(f"✅ Task evaluation works: score = {score:.4f}")

    except Exception as e:
        print(f"❌ Task evaluation test failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    print("\n3️⃣ Testing Math Agents...")
    try:
        from core import AgentManager

        config_with_task = EXPERIMENT_CONFIG.copy()
        config_with_task['task_name'] = 'gsm8k'

        agent_manager = AgentManager(config_with_task)
        agent_manager.initialize_agents("Test prompt")

        print(f"✅ Agent manager initialized with {len(agent_manager.agents)} agents")
        for i, agent in enumerate(agent_manager.agents):
            print(f"   Agent {i + 1}: {agent.agent_type} ({agent.task_name})")

    except Exception as e:
        print(f"❌ Agent initialization test failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    print("\n🎉 All minimal tests passed!")
    print("💡 Suggestion: Run the debug script next: python debug_gsm8k_setup.py")
    return True


if __name__ == "__main__":
    success = minimal_gsm8k_test()
    if success:
        print("\n✅ Basic GSM8k functionality works!")
    else:
        print("\n❌ Please fix the issues above")
