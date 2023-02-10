# Speaker experiments

This contains analyses of the behavior of our adaptive model in the speaker role

* `differentiation.py` examines how contrastive likelihood leads speaker to differentiate utterances over time (Appendix D, Fig. S3A)
* `speaker_adaptation_to_human_listener.py` evaluates offline ablations of the speaker (Appendix D, Fig. S3B)

* `/supplemental/` contains a series of supplemental analyses not reported in the paper:
  * `fix_cat_forgetting.py` examines effects of different KL regularization weights on reduction
  * `listener_accuracy.py` pairs speaker model w/ listener model to examine how they coordinate together
  * `get_fixed_points.py` records captions as a speaker adapts many times given the same image. Can be used to compute fixed points of reduction.
  * `hyperparam_sweep.py` examines speaker reduction across different values of hyperparameters (e.g. learning rate, batch size...)

