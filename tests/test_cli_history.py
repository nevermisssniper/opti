import numpy as np
from opti import cli


def test_demo_predictions_match_last_training_row(monkeypatch):
    original = cli.BehaviorOptimizer.recommend
    def recommend(self, today, target_perf, history):
        df, _ = cli.generate_synthetic_logs(cli.SyntheticDataConfig(n_days=60, seed=7))
        expected = self.features.transform(df).iloc[-1].to_numpy()
        actual = self.features.transform_single_with_history(today, history)
        np.testing.assert_allclose(actual, expected)
        return original(self, today, target_perf, history)
    monkeypatch.setattr(cli.BehaviorOptimizer, 'recommend', recommend)
    assert cli.run_demo(60, 7, {}, True) == 0
