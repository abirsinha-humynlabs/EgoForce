"""EgoForce evaluation and fusion stages for the egocentric hand-labelling pipeline.

Every module here is importable without CUDA, TensorRT, mmdet or mmpose. Those are only needed by
the two producer stages at the point where they actually load a model, so the topology, metrics,
matching and refit maths can be developed and tested on a laptop. ``fusion/selftest.py`` covers
exactly that surface.

See fusion/README.md for how the stages connect, and plan_of_action.md for what is still pending.
"""
