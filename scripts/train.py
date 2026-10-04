#!/usr/bin/env python3
"""CLI script to train TEC, AEC-Seq2seq, or Vanilla-Seq2seq models."""

import argparse
import os
import sys
from lingvo import compat as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tec import configs  # noqa: E402

_MODEL_MAP = {
    'TecSingleInterfering': configs.TecSingleInterfering,
    'TecMultiInterfering': configs.TecMultiInterfering,
    'NoSideInputSingleInterfering': configs.NoSideInputSingleInterfering,
    'NoSideInputMultiInterfering': configs.NoSideInputMultiInterfering,
    'AecSingleInterfering': configs.AecSingleInterfering,
    'AecMultiInterfering': configs.AecMultiInterfering,
}


def run_training(
    model_name: str = 'TecSingleInterfering',
    train_file_pattern: str = '',
    logdir: str = '/tmp/tec_train',
    max_steps: int = 10,
    batch_size: int = 2,
    learning_rate: float = 1e-3,
    save_interval_steps: int = 5,
):
  """Runs a training loop for the specified TEC model configuration."""
  if model_name not in _MODEL_MAP:
    raise ValueError(
        f'Unknown model_name {model_name}. Available: {list(_MODEL_MAP)}')

  os.makedirs(logdir, exist_ok=True)
  cfg = _MODEL_MAP[model_name]()
  inp_p = cfg.Train()
  inp_p.file_pattern = train_file_pattern
  inp_p.batch_size = batch_size
  inp_p.use_synthetic_data = not bool(train_file_pattern)

  task_p = cfg.Task()
  task_p.train.learning_rate = learning_rate

  losses = []
  with tf.Graph().as_default():
    inp = inp_p.Instantiate()
    task = task_p.Instantiate()

    batch = inp.GetPreprocessedInputBatch()
    predictions = task.ComputePredictions(task.theta, batch)
    metrics, _ = task.ComputeLoss(task.theta, predictions, batch)
    loss_tensor = metrics['loss'][0]

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    grads_and_vars = optimizer.compute_gradients(loss_tensor)
    clipped = [
        (tf.clip_by_norm(g, 1.0), v)
        for g, v in grads_and_vars
        if g is not None
    ]
    global_step = tf.train.get_or_create_global_step()
    train_op = optimizer.apply_gradients(clipped, global_step=global_step)
    saver = tf.train.Saver(max_to_keep=3)

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      for step_idx in range(1, max_steps + 1):
        _, loss_val, pre_val, post_val = sess.run([
            train_op,
            loss_tensor,
            metrics['loss_pre'][0],
            metrics['loss_post'][0],
        ])
        losses.append(float(loss_val))
        print(
            f'Step {step_idx}/{max_steps}: total_loss={loss_val:.4f} '
            f'(pre={pre_val:.4f}, post={post_val:.4f})')
        if step_idx % save_interval_steps == 0 or step_idx == max_steps:
          ckpt_path = os.path.join(logdir, 'model.ckpt')
          saver.save(sess, ckpt_path, global_step=step_idx)
  return losses


def main():
  parser = argparse.ArgumentParser(
      description='Train a Textual Echo Cancellation model.')
  parser.add_argument(
      '--model',
      type=str,
      default='TecSingleInterfering',
      choices=sorted(_MODEL_MAP.keys()),
      help='Model configuration name.')
  parser.add_argument(
      '--train_file_pattern',
      type=str,
      default='',
      help='Glob pattern of training TFRecord files. If empty, runs on '
      'synthetic data.')
  parser.add_argument(
      '--logdir',
      type=str,
      default='/tmp/tec_train',
      help='Directory for saving model checkpoints.')
  parser.add_argument(
      '--max_steps',
      type=int,
      default=100,
      help='Maximum number of training steps.')
  parser.add_argument(
      '--batch_size', type=int, default=4, help='Training batch size.')
  parser.add_argument(
      '--learning_rate',
      type=float,
      default=1e-3,
      help='Initial learning rate.')
  parser.add_argument(
      '--save_interval_steps',
      type=int,
      default=50,
      help='Checkpoint save frequency in steps.')
  args = parser.parse_args()

  run_training(
      model_name=args.model,
      train_file_pattern=args.train_file_pattern,
      logdir=args.logdir,
      max_steps=args.max_steps,
      batch_size=args.batch_size,
      learning_rate=args.learning_rate,
      save_interval_steps=args.save_interval_steps)


if __name__ == '__main__':
  main()
