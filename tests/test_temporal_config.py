"""Synthetic config fixtures: values here are not experimental defaults."""

import unittest
from argparse import Namespace
from cooperative.config import TemporalConfig


def synthetic_temporal():
    return dict(enabled=True, delay_ms=100, match_radius_m=1.0,
                match_weighting="inverse_distance", residual_hidden_dim=16,
                quality_hidden_dim=8, activation="relu", time_encoding="linear")


def config(temporal=None):
    return Namespace(npoints=2048, nblocks=2, input_dim=3, cpt=True,
                     temporal=synthetic_temporal() if temporal is None else temporal)


class ConfigTests(unittest.TestCase):
    def test_old_and_disabled_config_need_no_new_parameters(self):
        self.assertIsNone(TemporalConfig.from_config(Namespace(npoints=1)))
        self.assertIsNone(TemporalConfig.from_config({"temporal": {"enabled": False, "ignored": "old"}}))
        self.assertIsNone(TemporalConfig.from_config({"temporal": {}}))

    def test_each_unknown_setting_is_required(self):
        required = ("delay_ms", "match_radius_m", "match_weighting", "residual_hidden_dim",
                    "quality_hidden_dim", "activation", "time_encoding")
        for name in required:
            with self.subTest(name=name):
                settings = synthetic_temporal()
                del settings[name]
                with self.assertRaisesRegex(ValueError, name):
                    TemporalConfig.from_config(config(settings))

    def test_wandb_and_round_trip(self):
        original = TemporalConfig.from_config(config())
        wrapped = {name: {"value": value} for name, value in vars(config(original.as_dict())).items()}
        self.assertEqual(original, TemporalConfig.from_config(wrapped))
        described = {name: {'desc': None, 'value': value}
                     for name, value in vars(config(original.as_dict())).items()}
        self.assertEqual(original, TemporalConfig.from_config(described))
        # The online Agent passes the raw temporal W&B field through a Namespace.
        namespace = config({'desc': None, 'value': original.as_dict()})
        self.assertEqual(original, TemporalConfig.from_config(namespace))
        self.assertFalse(original.rr_gate)

    def test_architecture_dimensions_are_not_silently_changed(self):
        for field, value in (("npoints", 1024), ("nblocks", 3), ("input_dim", 4), ("cpt", False),
                             ("shared", True), ("earlyfusion", True)):
            with self.subTest(field=field):
                cfg = config()
                setattr(cfg, field, value)
                with self.assertRaisesRegex(ValueError, field):
                    TemporalConfig.from_config(cfg)

    def test_matching_temperature_only_required_for_softmax(self):
        settings = synthetic_temporal()
        settings["match_weighting"] = "softmax_squared_distance"
        with self.assertRaisesRegex(ValueError, "match_temperature"):
            TemporalConfig.from_config(config(settings))
        settings["match_temperature"] = 0.25
        self.assertEqual(TemporalConfig.from_config(config(settings)).match_temperature, 0.25)

    def test_unknown_or_fixed_paper_constants_rejected(self):
        settings = synthetic_temporal()
        settings["g_min"] = 0.4
        with self.assertRaisesRegex(ValueError, "g_min"):
            TemporalConfig.from_config(config(settings))

    def test_invalid_numbers_and_enums_fail(self):
        for field, value in (("delay_ms", 75), ("match_radius_m", float("nan")),
                             ("quality_hidden_dim", 0), ("residual_hidden_dim", True),
                             ("activation", "unknown"), ("time_encoding", "unknown"),
                             ("rr_gate", "false"), ("enabled", 1)):
            with self.subTest(field=field):
                settings = synthetic_temporal()
                settings[field] = value
                with self.assertRaises(ValueError):
                    TemporalConfig.from_config(config(settings))

    def test_training_hyperparameters_explicit(self):
        cfg = TemporalConfig.from_config(config())
        with self.assertRaisesRegex(ValueError, "beta.*sigma.*weight_decay.*epochs.*batch_size"):
            cfg.require_training()
        settings = synthetic_temporal()
        settings.update(beta=0.1, sigma=0.5, weight_decay=0.0, epochs=1, batch_size=2)
        cfg = TemporalConfig.from_config(config(settings))
        self.assertEqual(cfg.require_training()["batch_size"], 2)
        for field, bad in (("sigma", 0), ("beta", -1), ("epochs", 0), ("weight_decay", -1)):
            invalid = dict(settings, **{field: bad})
            with self.assertRaises(ValueError):
                TemporalConfig.from_config(config(invalid))


if __name__ == "__main__":
    unittest.main()
