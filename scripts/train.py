#!/usr/bin/env python3
"""CLI script to train TEC, AEC-Seq2seq, or Vanilla-Seq2seq models."""

import argparse
import os
import sys
import time
from absl import app
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
    learning_rate: float = 1e-4,
    decay_steps: int = 50000,
    decay_rate: float = 0.1,
    save_interval_steps: int = 5,
    restore_checkpoint: str = '',
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

    global_step = tf.train.get_or_create_global_step()
    step_float = tf.cast(global_step, tf.float32)
    decayed_lr = learning_rate * tf.pow(
        float(decay_rate), step_float / float(max(1, decay_steps)))
    lr_tensor = tf.maximum(decayed_lr, learning_rate * decay_rate)

    optimizer = tf.train.AdamOptimizer(
        learning_rate=lr_tensor, beta1=0.9, beta2=0.999, epsilon=1e-6)
    grads_and_vars = optimizer.compute_gradients(loss_tensor)
    clipped = [
        (tf.clip_by_norm(g, 1.0), v)
        for g, v in grads_and_vars
        if g is not None
    ]
    train_op = optimizer.apply_gradients(clipped, global_step=global_step)
    saver = tf.train.Saver(max_to_keep=3)
    best_saver = tf.train.Saver(max_to_keep=1)
    best_loss = float('inf')

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      if restore_checkpoint:
        saver.restore(sess, restore_checkpoint)
      t0 = time.time()
      for step_idx in range(1, max_steps + 1):
        _, loss_val, pre_val, post_val, lr_val = sess.run([
            train_op,
            loss_tensor,
            metrics['loss_pre'][0],
            metrics['loss_post'][0],
            lr_tensor,
        ])
        losses.append(float(loss_val))
        elapsed = time.time() - t0
        print(
            f'Step {step_idx}/{max_steps} ({elapsed:.1f}s, lr={lr_val:.2e}): '
            f'total_loss={loss_val:.4f} (pre={pre_val:.4f}, '
            f'post={post_val:.4f})',
            flush=True)
        if float(loss_val) < best_loss:
          best_loss = float(loss_val)
          best_saver.save(sess, os.path.join(logdir, 'best.ckpt'))
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
      default=1e-4,
      help='Initial learning rate (default 1e-4 per Section 3.4 of paper).')
  parser.add_argument(
      '--decay_steps',
      type=int,
      default=50000,
      help='Exponential decay steps (default 50000 per Section 3.4 of paper).')
  parser.add_argument(
      '--save_interval_steps',
      type=int,
      default=50,
      help='Checkpoint save frequency in steps.')
  parser.add_argument(
      '--restore_checkpoint',
      type=str,
      default='',
      help='Optional checkpoint path to restore before training.')
  args = parser.parse_args()

  run_training(
      model_name=args.model,
      train_file_pattern=args.train_file_pattern,
      logdir=args.logdir,
      max_steps=args.max_steps,
      batch_size=args.batch_size,
      learning_rate=args.learning_rate,
      decay_steps=args.decay_steps,
      save_interval_steps=args.save_interval_steps,
      restore_checkpoint=args.restore_checkpoint)


if __name__ == '__main__':
  app.run(lambda _: main(), argv=sys.argv[:1])
