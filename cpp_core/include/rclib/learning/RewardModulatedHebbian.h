#pragma once

#include <Eigen/Dense>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

/// Configuration of a reward-modulated Hebbian rule. A component (readout or reservoir) adds
/// exploration noise to the potentials of its post-synaptic units, correlates pre-synaptic
/// activity with a post-synaptic signal into an eligibility, and changes its weights in
/// proportion to the eligibility and a reward-derived modulator:
///
///   pre       the row that multiplies W to form the potential (row-vector convention)
///   post      NOISE: the injected noise; HIGH_PASS: v - v_avg, the potential minus its
///             average before this row; POTENTIAL: the potential v itself
///   S(y)      sign(y) |y|^eligibility_power, so S(ab) = S(a) S(b)
///   ONLINE    e <- trace_decay * e + S(pre)^T S(post) every exploring row
///   TRIAL     E <- E + S(pre)^T S(post) every exploring row (and C <- C + pre^T pre when
///             decorrelating), cleared by every update
///   update    dW = eta_n * m * G, G = e, E, or (C + decorrelation * I)^-1 E, each entry
///             clipped to [-max_update, max_update]
///   m         DIFFERENCE: R - R_avg; HEAVISIDE: 1 if R > R_avg, else 0
///   eta_n     learning_rate / (1 + n / learning_rate_decay_steps), n successful updates so far
///
/// The weights have shape (n_pre, n_post). The papers whose rules these settings express are
/// listed in docs/theory/reward_modulated_hebbian.md.
struct RewardModulatedHebbianRule {
  /// Distribution of the exploration noise: UNIFORM on [-amplitude, amplitude], or GAUSSIAN
  /// with standard deviation amplitude. The underlying types are fixed, so a cast from any
  /// other int is well defined; validate() rejects it.
  enum Distribution : int { UNIFORM, GAUSSIAN };
  enum PostSignal : int { NOISE, HIGH_PASS, POTENTIAL };
  enum Update : int { ONLINE, TRIAL };
  enum Modulator : int { DIFFERENCE, HEAVISIDE };
  /// The reward baseline R_avg, kept per integer key: EMA, R_avg <- baseline_decay * R_avg +
  /// (1 - baseline_decay) * R; WINDOW, the mean of the last baseline_window rewards; NONE, 0.
  enum Baseline : int { EMA, WINDOW, NONE };

  double learning_rate = 1e-3;
  Distribution noise_distribution = UNIFORM;
  double noise_amplitude = 0.1;
  /// Probability that a unit is perturbed in a row; 1 perturbs every unit in every row.
  double noise_probability = 1.0;
  unsigned int noise_seed = 0;
  PostSignal post = HIGH_PASS;
  /// Decay of the post-synaptic average for HIGH_PASS; 0 makes the post signal the first difference.
  double post_decay = 0.0;
  double eligibility_power = 1.0;
  Update update = ONLINE;
  /// Decay of the ONLINE eligibility per exploring row; TRIAL eligibility never decays.
  double trace_decay = 0.0;
  Modulator modulator = DIFFERENCE;
  Baseline baseline = EMA;
  double baseline_decay = 0.8;
  int baseline_window = 50;
  std::optional<double> learning_rate_decay_steps;
  std::optional<double> max_update;
  /// Ridge term lambda of the decorrelated TRIAL update; requires eligibility_power 1.
  std::optional<double> decorrelation;

  /// Throws std::invalid_argument naming the first invalid setting. NaN fails every check.
  void validate() const;
};

/// SplitMix64 (Steele, Lea and Flood 2014): one u64 of state, so a saved model continues the
/// exact noise sequence.
class SplitMix64 {
public:
  explicit SplitMix64(std::uint64_t state = 0) : state(state) {}
  std::uint64_t next();
  /// Uniform on [0, 1) from the top 53 bits of next().
  double uniform();
  std::uint64_t getState() const { return state; }
  void setState(std::uint64_t value) { state = value; }

private:
  std::uint64_t state;
};

/// The learning state of one reward-modulated component: exploration noise, the post-synaptic
/// average, the eligibility, the per-key reward baselines and the update count. The component
/// owns the weights; the engine computes their changes.
///
/// Rows and updates each go through two phases, so that a model with several components can
/// apply a row or an update to all of them or to none: stage or prepare computes and checks
/// everything without changing the learning state, then commit applies it or discard drops it.
class RewardModulatedHebbian {
public:
  /// Validates the rule. `stream_salt` separates the noise of components that share a rule and
  /// seed; `context` names the component in error messages.
  RewardModulatedHebbian(const RewardModulatedHebbianRule &rule, std::uint64_t stream_salt, std::string context);

  const RewardModulatedHebbianRule &getRule() const { return rule; }
  bool isInitialized() const { return initialized; }
  Eigen::Index getNPre() const { return eligibility.rows(); }
  Eigen::Index getNPost() const { return eligibility.cols(); }
  /// Allocates zero eligibility (and covariance) for weights of shape (n_pre, n_post). Once only.
  void initialize(Eigen::Index n_pre, Eigen::Index n_post);

  /// Draws the exploration noise of one row of n_post units, consuming a fixed number of draws
  /// per unit: one for the perturbation gate, then one (UNIFORM) or two (GAUSSIAN) for the value.
  Eigen::RowVectorXd sampleNoise();

  /// Computes the contribution of one exploring row and checks that the potential, the post
  /// signal, S(pre), S(post) and the bounds on the new eligibility and covariance are finite.
  /// Throws std::runtime_error naming the component otherwise. Changes no learning state.
  void stageRow(const Eigen::RowVectorXd &pre, const Eigen::RowVectorXd &potential, const Eigen::RowVectorXd &noise);
  /// Applies the staged row: eligibility, covariance, post-synaptic average, bounds and the
  /// has-eligibility flag. Throws std::logic_error if no row is staged.
  void commitRow();
  void discardRow() { staged_row.reset(); }

  /// Throws std::logic_error unless an exploring row was committed since the last update.
  void checkReady() const;
  /// Prepares the update for a reward. Throws std::invalid_argument for a non-finite reward or
  /// baseline, std::logic_error if not ready, and std::runtime_error if the modulator or the
  /// weight change is not finite or the decorrelation solve fails. Changes no learning state.
  void prepareReinforce(double reward, int key, std::optional<double> baseline);
  /// Prepares an update with the modulator m given directly; the baselines are not used.
  void prepareModulate(double modulator);
  /// The prepared weight change, or nullptr when the modulator is zero (nothing changes).
  const Eigen::MatrixXd *preparedDelta() const;
  /// Applies the prepared update: the baseline of its key, the update count, and in TRIAL mode
  /// the clearing of the eligibility and covariance. The component adds preparedDelta() to its
  /// weights. Throws std::logic_error if no update is prepared.
  void commitUpdate();
  void discardUpdate() { prepared_update.reset(); }

  /// Forgets the post-synaptic average, so the next exploring row has a zero HIGH_PASS signal.
  void resetActivity() { filter_initialized = false; }
  /// Zeroes the eligibility and covariance and their bounds and clears the has-eligibility flag.
  void clearEligibility();

  // Read-only learning state.
  const Eigen::MatrixXd &getEligibility() const { return eligibility; }
  const Eigen::MatrixXd &getCovariance() const { return covariance; }
  double getEligibilityBound() const { return eligibility_bound; }
  double getCovarianceBound() const { return covariance_bound; }
  const Eigen::RowVectorXd &getPostAverage() const { return post_average; }
  bool isPostAverageInitialized() const { return filter_initialized; }
  bool hasEligibility() const { return has_eligibility; }
  std::uint64_t getUpdateCount() const { return update_count; }
  std::uint64_t getNoiseState() const { return rng.getState(); }
  /// The reward history of each key: the EMA value (one entry) or the window, oldest first.
  const std::map<int, std::vector<double>> &getBaselines() const { return baselines; }
  /// The baseline the next reward of `key` is compared with, or nothing before its first reward.
  std::optional<double> getBaseline(int key) const;
  /// The learning rate the next update uses.
  double getLearningRate() const;

  /// S(y) = sign(y) |y|^power.
  static double eligibilityFunction(double y, double power);

private:
  struct StagedRow {
    Eigen::RowVectorXd pre;
    Eigen::RowVectorXd s_pre;
    Eigen::RowVectorXd s_post;
    Eigen::RowVectorXd post_average;
    double eligibility_bound;
    double covariance_bound;
  };
  struct PreparedUpdate {
    std::optional<Eigen::MatrixXd> delta; // empty when the modulator is zero
    std::optional<int> key;               // the baseline to replace, if any
    std::vector<double> baseline_history;
  };

  void prepare(double modulator, std::optional<int> key, std::vector<double> history);
  void fail(const std::string &what) const;

  RewardModulatedHebbianRule rule;
  std::string context;
  SplitMix64 rng;
  bool initialized = false;

  Eigen::MatrixXd eligibility; // (n_pre, n_post)
  Eigen::MatrixXd covariance;  // (n_pre, n_pre) with decorrelation, else empty
  double eligibility_bound = 0.0;
  double covariance_bound = 0.0;
  Eigen::RowVectorXd post_average;
  bool filter_initialized = false;
  bool has_eligibility = false;
  std::uint64_t update_count = 0;
  std::map<int, std::vector<double>> baselines;

  std::optional<StagedRow> staged_row;
  std::optional<PreparedUpdate> prepared_update;
};
