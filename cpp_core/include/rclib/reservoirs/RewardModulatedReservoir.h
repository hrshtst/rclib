#pragma once

#include "rclib/Reservoir.h"
#include "rclib/learning/RewardModulatedHebbian.h"
#include "rclib/learning/RewardModulatedLearner.h"
#include "rclib/reservoirs/RandomSparseReservoir.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <optional>
#include <vector>

/// A random sparse reservoir whose recurrent weights learn from rewards. Its neurons integrate
/// a potential v and fire at the rate r = tanh(v), with the state and input as row vectors:
///
///   v = (1 - leak_rate) * v + leak_rate * (input * W_in + bias + r_prev * W_res) + noise
///
/// where the exploration noise of the rule enters only exploring steps, outside the leak. This
/// potential leak, unlike RandomSparseReservoir's leak on the rate, keeps a perturbation in the
/// dynamics for about 1 / leak_rate steps. With leak_rate 1 and no noise the update equals
/// RandomSparseReservoir's, and the same hyperparameters and seed draw the same W_res, bias and
/// W_in. Learning uses the previous rate r_prev as the pre-synaptic activity and the potential
/// as the post-synaptic one; it changes only existing synapses (the sparsity pattern stays) from
/// a plastic pre-synaptic to a plastic post-synaptic neuron.
class RewardModulatedReservoir : public Reservoir, public RewardModulatedLearner {
public:
  /// The first nine parameters are those of RandomSparseReservoir. `plastic_presynaptic` and
  /// `plastic_postsynaptic` list the neurons whose outgoing and incoming synapses learn; nothing
  /// means all of them. Throws std::invalid_argument for an invalid setting or neuron index.
  RewardModulatedReservoir(int n_neurons, double spectral_radius, double sparsity, double leak_rate,
                           double input_scaling, bool include_bias, unsigned int seed,
                           RandomSparseReservoir::SpectralRadiusMethod spectral_radius_method, double bias_scaling,
                           const RewardModulatedHebbianRule &rule,
                           std::optional<std::vector<int>> plastic_presynaptic = std::nullopt,
                           std::optional<std::vector<int>> plastic_postsynaptic = std::nullopt);

  /// A noise-free step that changes no learning state.
  const Eigen::MatrixXd &advance(const Eigen::MatrixXd &input) override;
  /// Resets the rates and potentials to zero and forgets the post-synaptic average. The weights
  /// and the learning state stay.
  void resetState() override;
  const Eigen::MatrixXd &getState() const override { return state; }
  int getOutputDim(int /*input_dim*/) const override { return n_neurons; }
  int getInputDim() const override { return W_in_initialized ? static_cast<int>(W_in.rows()) : 0; }
  /// A step of Model::act(): exploring adds noise and stages the step's eligibility.
  const Eigen::MatrixXd &stageAct(const Eigen::MatrixXd &input, bool explore) override;

  /// Steps through the input rows in order and returns the rates of each. Exploring adds noise
  /// and records the eligibility; a noise-free step changes no learning state. Rows must be
  /// finite (std::invalid_argument, nothing changes); a row whose potential or eligibility is
  /// not finite throws std::runtime_error, changing no learning state, and the activity is then
  /// unspecified until resetState().
  Eigen::MatrixXd act(const Eigen::MatrixXd &inputs, bool explore = true);
  /// Explores with the given noise, (rows, n_neurons), instead of drawing it.
  Eigen::MatrixXd act(const Eigen::MatrixXd &inputs, const Eigen::MatrixXd &noise);

  int getNNeurons() const { return n_neurons; }
  double getSpectralRadius() const { return spectral_radius; }
  double getSparsity() const { return sparsity; }
  double getLeakRate() const { return leak_rate; }
  double getInputScaling() const { return input_scaling; }
  bool getIncludeBias() const { return include_bias; }
  unsigned int getSeed() const { return seed; }
  RandomSparseReservoir::SpectralRadiusMethod getSpectralRadiusMethod() const { return spectral_radius_method; }
  double getBiasScaling() const { return bias_scaling; }
  const std::optional<std::vector<int>> &getPlasticPresynaptic() const { return plastic_presynaptic; }
  const std::optional<std::vector<int>> &getPlasticPostsynaptic() const { return plastic_postsynaptic; }
  const RewardModulatedHebbianRule &getRule() const { return learning.getRule(); }
  const RewardModulatedHebbian &getLearning() const { return learning; }

  /// The potentials v after the last step, (1, n_neurons) like the state.
  const Eigen::MatrixXd &getPotential() const { return potential; }
  /// The recurrent weights, (n_neurons, n_neurons), entry (i, j) from neuron i to neuron j. They
  /// start scaled to the spectral radius and change as the reservoir learns.
  const Eigen::SparseMatrix<double> &getRecurrentWeights() const { return W_res; }
  /// The input weights, (input_dim, n_neurons), drawn at the first step; throws before it.
  const Eigen::MatrixXd &getInputWeights() const;
  const Eigen::RowVectorXd &getBias() const { return bias; }

  void commitRow() override { learning.commitRow(); }
  void discardRow() override { learning.discardRow(); }
  void checkReady() const override { learning.checkReady(); }
  void prepareReinforce(double reward, int key, std::optional<double> baseline) override;
  void prepareModulate(double modulator) override;
  void commitUpdate() override;
  void discardUpdate() override;
  void clearEligibility() override { learning.clearEligibility(); }
  void resetActivity() override { learning.resetActivity(); }

private:
  void validateParameters() const;
  void checkInputs(const Eigen::MatrixXd &inputs) const;
  /// Computes the potential and rates of one row: noise-free, or with `noise` staging the row.
  void step(const Eigen::MatrixXd &input, const Eigen::RowVectorXd *noise);
  void prepareWeights();

  int n_neurons;
  double spectral_radius;
  double sparsity;
  double leak_rate;
  double input_scaling;
  bool include_bias;
  unsigned int seed;
  RandomSparseReservoir::SpectralRadiusMethod spectral_radius_method;
  double bias_scaling;
  std::optional<std::vector<int>> plastic_presynaptic;  // sorted; nothing means all
  std::optional<std::vector<int>> plastic_postsynaptic; // sorted; nothing means all
  RewardModulatedHebbian learning;

  Eigen::MatrixXd state;     // rates r, (1, n_neurons)
  Eigen::MatrixXd potential; // v, (1, n_neurons), laid out as the state
  Eigen::MatrixXd drive;     // input * W_in + bias + r_prev * W_res, pre-allocated
  Eigen::SparseMatrix<double> W_res;
  Eigen::MatrixXd W_in;
  Eigen::RowVectorXd bias;
  bool W_in_initialized = false;
  std::optional<Eigen::VectorXd> prepared_values; // new W_res values, in storage order
};
