#pragma once

#include "rclib/Reservoir.h"

#include <Eigen/Sparse>
#include <memory>

class BinaryReader;
class BinaryWriter;

class RandomSparseReservoir : public Reservoir {
public:
  /// How the constructor finds the spectral radius of the random W_res before scaling it to
  /// spectral_radius. POWER_ITERATION estimates it by seeded power iteration on the sparse
  /// matrix, typically to within 0.3%. DENSE takes the largest eigenvalue modulus of W_res
  /// as a dense matrix, which is exact but costs O(n_neurons^3) time and O(n_neurons^2)
  /// memory, so it is meant for small reservoirs. The underlying type is fixed, so a cast from
  /// any other int is well defined; the constructor rejects it.
  enum SpectralRadiusMethod : int { POWER_ITERATION, DENSE };

  RandomSparseReservoir(int n_neurons, double spectral_radius, double sparsity = 0.1, double leak_rate = 1.0,
                        double input_scaling = 1.0, bool include_bias = false, unsigned int seed = 42,
                        SpectralRadiusMethod spectral_radius_method = POWER_ITERATION);

  const Eigen::MatrixXd &advance(const Eigen::MatrixXd &input) override;
  void resetState() override;
  const Eigen::MatrixXd &getState() const override;
  int getOutputDim(int input_dim) const override;
  int getInputDim() const override { return W_in_initialized ? static_cast<int>(W_in.rows()) : 0; }

  int getNNeurons() const { return n_neurons; }
  double getSpectralRadius() const { return spectral_radius; }
  double getSparsity() const { return sparsity; }
  double getLeakRate() const { return leak_rate; }
  double getInputScaling() const { return input_scaling; }
  bool getIncludeBias() const { return include_bias; }
  unsigned int getSeed() const { return seed; }
  SpectralRadiusMethod getSpectralRadiusMethod() const { return spectral_radius_method; }

  /// Writes the hyperparameters and the full state (weights and current activations)
  /// in the model file format; the type tag is written by Model. Throws SerializationError.
  void save(BinaryWriter &writer) const;
  /// Reads a payload written by save(). Throws SerializationError on invalid input.
  static std::shared_ptr<RandomSparseReservoir> load(BinaryReader &reader);
  /// Checks the hyperparameters and that the stored matrices have consistent shapes.
  /// Saving and loading both run it. Throws SerializationError.
  void checkConsistency() const;

private:
  RandomSparseReservoir() = default; // used by load(): skips generating the weights
  void validateParameters() const;
  void initialize_W_in(int input_dim);

  int n_neurons;
  double spectral_radius;
  double sparsity;
  double leak_rate;
  double input_scaling; // New member variable
  bool include_bias;
  unsigned int seed;
  SpectralRadiusMethod spectral_radius_method;
  bool W_in_initialized;

  Eigen::MatrixXd state;
  Eigen::MatrixXd temp_state; // Pre-allocated temporary
  Eigen::SparseMatrix<double> W_res;
  Eigen::MatrixXd W_in;
  Eigen::RowVectorXd bias;
};
