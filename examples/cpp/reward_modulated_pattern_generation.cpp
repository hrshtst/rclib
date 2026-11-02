// Learns to generate a sine wave from a reward signal alone, after Hoerzer et al. (2014): a
// chaotic reservoir receives its own noisy output as input, and a reward-modulated readout learns
// only whether each step's squared error beat its recent average. See the Python example of the
// same name for plots and a run without exploration noise.

#include "rclib/Model.h"
#include "rclib/readouts/RewardModulatedReadout.h"
#include "rclib/reservoirs/RandomSparseReservoir.h"

#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <memory>

namespace {

constexpr double pi = 3.14159265358979323846;
constexpr double period = 100.0;

double target(int step) { return 0.5 * std::sin(2.0 * pi * step / period); }

// The error of a noise-free continuation, over the target's variance (0.125).
double normalizedError(Model &model, int start, int steps) {
  const Eigen::MatrixXd generated = model.predictGenerative(Eigen::MatrixXd(0, 1), steps);
  double sum = 0.0;
  for (int i = 0; i < steps; ++i) {
    const double error = generated(i, 0) - target(start + i);
    sum += error * error;
  }
  return sum / steps / 0.125;
}

} // namespace

int main() {
  // The Hoerzer et al. variant: first difference of the noisy output as the post-synaptic signal,
  // a Heaviside modulator against an exponential reward average, and a decaying learning rate.
  RewardModulatedHebbianRule rule;
  rule.learning_rate = 5e-3;
  rule.learning_rate_decay_steps = 10000.0;
  rule.noise_amplitude = 0.2;
  rule.post = RewardModulatedHebbianRule::HIGH_PASS;
  rule.post_decay = 0.0;
  rule.modulator = RewardModulatedHebbianRule::HEAVISIDE;
  rule.baseline_decay = 0.8;

  // The input is the fed-back output, so the input weights are the feedback weights.
  Model model;
  model.addReservoir(std::make_shared<RandomSparseReservoir>(300, 1.2, 0.1, 0.1, 1.0, false, 0));
  model.setReadout(std::make_shared<RewardModulatedReadout>(1, rule, false));

  const int train_steps = 100000;
  const int window = 1000;
  Eigen::MatrixXd output = Eigen::MatrixXd::Zero(1, 1);
  double first_error = 0.0;
  double last_error = 0.0;
  for (int step = 0; step < train_steps; ++step) {
    output = model.act(output); // the noisy output is fed back as the next input
    const double error = output(0, 0) - target(step);
    model.reinforce(-error * error);
    if (step < window) {
      first_error += error * error / window;
    } else if (step >= train_steps - window) {
      last_error += error * error / window;
    }
  }
  std::cout << "Mean squared error of the noisy output, first " << window << " steps: " << first_error << ", last "
            << window << " steps: " << last_error << "\n";
  std::cout << "Generated without noise after training: normalized error " << normalizedError(model, train_steps, 500)
            << "\n";
  return 0;
}
