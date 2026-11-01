#include "rclib/learning/RewardModulatedHebbian.h"

#include <Eigen/Dense>
#include <catch2/catch_all.hpp>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <vector>

namespace {

using Rule = RewardModulatedHebbianRule;

constexpr double nan_value = std::numeric_limits<double>::quiet_NaN();
constexpr double inf_value = std::numeric_limits<double>::infinity();
constexpr std::uint64_t golden_gamma = 0x9E3779B97F4A7C15ULL;

Eigen::RowVectorXd row(std::initializer_list<double> values) {
  Eigen::RowVectorXd result(static_cast<Eigen::Index>(values.size()));
  Eigen::Index i = 0;
  for (double value : values) {
    result(i++) = value;
  }
  return result;
}

// Everything the engine learns, for checking that a rejected call changed nothing.
struct Snapshot {
  Eigen::MatrixXd eligibility;
  Eigen::MatrixXd covariance;
  double eligibility_bound;
  double covariance_bound;
  Eigen::RowVectorXd post_average;
  bool post_average_initialized;
  bool has_eligibility;
  std::uint64_t update_count;
  std::map<int, std::vector<double>> baselines;

  explicit Snapshot(const RewardModulatedHebbian &engine)
      : eligibility(engine.getEligibility()), covariance(engine.getCovariance()),
        eligibility_bound(engine.getEligibilityBound()), covariance_bound(engine.getCovarianceBound()),
        post_average(engine.getPostAverage()), post_average_initialized(engine.isPostAverageInitialized()),
        has_eligibility(engine.hasEligibility()), update_count(engine.getUpdateCount()),
        baselines(engine.getBaselines()) {}

  bool operator==(const Snapshot &other) const {
    return eligibility == other.eligibility && covariance == other.covariance &&
           eligibility_bound == other.eligibility_bound && covariance_bound == other.covariance_bound &&
           post_average == other.post_average && post_average_initialized == other.post_average_initialized &&
           has_eligibility == other.has_eligibility && update_count == other.update_count &&
           baselines == other.baselines;
  }
};

RewardModulatedHebbian makeEngine(const Rule &rule, Eigen::Index n_pre, Eigen::Index n_post) {
  RewardModulatedHebbian engine(rule, 0, "Engine");
  engine.initialize(n_pre, n_post);
  return engine;
}

void commitRow(RewardModulatedHebbian &engine, const Eigen::RowVectorXd &pre, const Eigen::RowVectorXd &potential,
               const Eigen::RowVectorXd &noise) {
  engine.stageRow(pre, potential, noise);
  engine.commitRow();
}

void commitRow(RewardModulatedHebbian &engine, const Eigen::RowVectorXd &pre, const Eigen::RowVectorXd &potential) {
  commitRow(engine, pre, potential, Eigen::RowVectorXd::Zero(potential.size()));
}

// Prepares and commits a reward; returns the weight change, or a zero matrix for a zero modulator.
Eigen::MatrixXd reinforce(RewardModulatedHebbian &engine, double reward, int key = 0,
                          std::optional<double> baseline = std::nullopt) {
  engine.prepareReinforce(reward, key, baseline);
  const Eigen::MatrixXd *delta = engine.preparedDelta();
  Eigen::MatrixXd result =
      delta == nullptr ? Eigen::MatrixXd::Zero(engine.getNPre(), engine.getNPost()) : Eigen::MatrixXd(*delta);
  engine.commitUpdate();
  return result;
}

} // namespace

TEST_CASE("RewardModulatedHebbianRule - validate accepts the defaults and rejects each invalid setting",
          "[RewardModulatedHebbian]") {
  Rule rule;
  REQUIRE_NOTHROW(rule.validate());

  const double bad = GENERATE(nan_value, inf_value, -1.0);
  SECTION("learning_rate") { rule.learning_rate = bad; }
  SECTION("noise_amplitude") { rule.noise_amplitude = bad; }
  SECTION("noise_probability") { rule.noise_probability = bad; }
  SECTION("post_decay") { rule.post_decay = bad; }
  SECTION("eligibility_power") { rule.eligibility_power = bad; }
  SECTION("trace_decay") { rule.trace_decay = bad; }
  SECTION("baseline_decay") { rule.baseline_decay = bad; }
  SECTION("learning_rate_decay_steps") { rule.learning_rate_decay_steps = bad; }
  SECTION("max_update") { rule.max_update = bad; }
  SECTION("decorrelation") {
    rule.update = Rule::TRIAL;
    rule.decorrelation = bad;
  }
  REQUIRE_THROWS_AS(rule.validate(), std::invalid_argument);
}

TEST_CASE("RewardModulatedHebbianRule - validate rejects out-of-range values and combinations",
          "[RewardModulatedHebbian]") {
  Rule rule;
  SECTION("zero learning rate") { rule.learning_rate = 0.0; }
  SECTION("zero noise probability") { rule.noise_probability = 0.0; }
  SECTION("noise probability above 1") { rule.noise_probability = 1.5; }
  SECTION("post_decay of 1") { rule.post_decay = 1.0; }
  SECTION("baseline_decay of 1") { rule.baseline_decay = 1.0; }
  SECTION("trace_decay above 1") { rule.trace_decay = 1.5; }
  SECTION("zero eligibility power") { rule.eligibility_power = 0.0; }
  SECTION("empty baseline window") { rule.baseline_window = 0; }
  SECTION("huge baseline window") { rule.baseline_window = 1000001; }
  SECTION("unknown enum values") {
    const int which = GENERATE(0, 1, 2, 3, 4);
    switch (which) {
      case 0:
        rule.noise_distribution = static_cast<Rule::Distribution>(7);
        break;
      case 1:
        rule.post = static_cast<Rule::PostSignal>(7);
        break;
      case 2:
        rule.update = static_cast<Rule::Update>(7);
        break;
      case 3:
        rule.modulator = static_cast<Rule::Modulator>(7);
        break;
      default:
        rule.baseline = static_cast<Rule::Baseline>(7);
        break;
    }
  }
  SECTION("trace decay with trial updates") {
    rule.update = Rule::TRIAL;
    rule.trace_decay = 0.5;
  }
  SECTION("decorrelation with online updates") { rule.decorrelation = 1.0; }
  SECTION("decorrelation with a supralinear eligibility") {
    rule.update = Rule::TRIAL;
    rule.decorrelation = 1.0;
    rule.eligibility_power = 3.0;
  }
  REQUIRE_THROWS_AS(rule.validate(), std::invalid_argument);
  REQUIRE_THROWS_AS(RewardModulatedHebbian(rule, 0, "Engine"), std::invalid_argument);
}

TEST_CASE("SplitMix64 - gives the reference sequence", "[RewardModulatedHebbian]") {
  // Reference outputs of Vigna's splitmix64.c.
  SplitMix64 generator(1234567);
  const std::vector<std::uint64_t> expected = {6457827717110365317ULL, 3203168211198807973ULL, 9817491932198370423ULL,
                                               4593380528125082431ULL, 16408922859458223821ULL};
  for (std::uint64_t value : expected) {
    REQUIRE(generator.next() == value);
  }
  SplitMix64 zero(0);
  REQUIRE(zero.next() == 0xE220A8397B1DCDAFULL);
  REQUIRE(zero.next() == 0x6E789E6AA1B965F4ULL);

  SplitMix64 unit(42);
  for (int i = 0; i < 1000; ++i) {
    const double u = unit.uniform();
    REQUIRE(u >= 0.0);
    REQUIRE(u < 1.0);
  }
}

TEST_CASE("RewardModulatedHebbian - noise takes a fixed number of draws per unit", "[RewardModulatedHebbian]") {
  Rule rule;
  rule.noise_distribution = GENERATE(Rule::UNIFORM, Rule::GAUSSIAN);
  rule.noise_probability = GENERATE(1.0, 0.3);
  RewardModulatedHebbian engine = makeEngine(rule, 2, 5);
  const std::uint64_t draws_per_unit = rule.noise_distribution == Rule::UNIFORM ? 2 : 3;
  for (int step = 0; step < 4; ++step) {
    const std::uint64_t before = engine.getNoiseState();
    engine.sampleNoise();
    // Every draw adds the Weyl increment to the state.
    REQUIRE(engine.getNoiseState() - before == 5 * draws_per_unit * golden_gamma);
  }
}

TEST_CASE("RewardModulatedHebbian - noise has the configured distribution and rate", "[RewardModulatedHebbian]") {
  Rule rule;
  rule.noise_amplitude = 0.5;
  const int n_units = 2000;
  const int n_rows = 50;

  SECTION("uniform") {
    rule.noise_distribution = Rule::UNIFORM;
    RewardModulatedHebbian engine = makeEngine(rule, 1, n_units);
    double sum = 0.0;
    double squares = 0.0;
    for (int r = 0; r < n_rows; ++r) {
      const Eigen::RowVectorXd noise = engine.sampleNoise();
      REQUIRE(noise.cwiseAbs().maxCoeff() <= 0.5);
      sum += noise.sum();
      squares += noise.squaredNorm();
    }
    const double count = static_cast<double>(n_units) * n_rows;
    REQUIRE_THAT(sum / count, Catch::Matchers::WithinAbs(0.0, 0.01));
    REQUIRE_THAT(squares / count, Catch::Matchers::WithinRel(0.25 / 3.0, 0.03));
  }
  SECTION("gaussian") {
    rule.noise_distribution = Rule::GAUSSIAN;
    RewardModulatedHebbian engine = makeEngine(rule, 1, n_units);
    double sum = 0.0;
    double squares = 0.0;
    for (int r = 0; r < n_rows; ++r) {
      const Eigen::RowVectorXd noise = engine.sampleNoise();
      REQUIRE(noise.allFinite());
      sum += noise.sum();
      squares += noise.squaredNorm();
    }
    const double count = static_cast<double>(n_units) * n_rows;
    REQUIRE_THAT(sum / count, Catch::Matchers::WithinAbs(0.0, 0.01));
    REQUIRE_THAT(squares / count, Catch::Matchers::WithinRel(0.25, 0.03));
  }
  SECTION("sparse perturbations") {
    rule.noise_probability = 0.1;
    RewardModulatedHebbian engine = makeEngine(rule, 1, n_units);
    double perturbed = 0.0;
    for (int r = 0; r < n_rows; ++r) {
      perturbed += static_cast<double>((engine.sampleNoise().array() != 0.0).count());
    }
    REQUIRE_THAT(perturbed / (static_cast<double>(n_units) * n_rows), Catch::Matchers::WithinAbs(0.1, 0.005));
  }
}

TEST_CASE("RewardModulatedHebbian - the stream depends on the seed and the salt", "[RewardModulatedHebbian]") {
  Rule rule;
  RewardModulatedHebbian a(rule, 1, "A");
  RewardModulatedHebbian b(rule, 1, "B");
  RewardModulatedHebbian c(rule, 2, "C");
  rule.noise_seed = 7;
  RewardModulatedHebbian d(rule, 1, "D");
  for (RewardModulatedHebbian *engine : {&a, &b, &c, &d}) {
    engine->initialize(1, 8);
  }
  const Eigen::RowVectorXd first = a.sampleNoise();
  REQUIRE(b.sampleNoise() == first);
  REQUIRE(c.sampleNoise() != first);
  REQUIRE(d.sampleNoise() != first);
}

TEST_CASE("RewardModulatedHebbian - the eligibility function is multiplicative", "[RewardModulatedHebbian]") {
  const double power = GENERATE(1.0, 1.5, 2.0, 3.0);
  const std::vector<double> values = {-2.5, -0.3, 0.0, 0.7, 1.9};
  for (double a : values) {
    for (double b : values) {
      const double product = RewardModulatedHebbian::eligibilityFunction(a * b, power);
      const double factored =
          RewardModulatedHebbian::eligibilityFunction(a, power) * RewardModulatedHebbian::eligibilityFunction(b, power);
      REQUIRE_THAT(product, Catch::Matchers::WithinAbs(factored, 1e-12));
    }
  }
  REQUIRE(RewardModulatedHebbian::eligibilityFunction(-2.0, 3.0) == -8.0);
  REQUIRE(RewardModulatedHebbian::eligibilityFunction(-2.0, 2.0) == -4.0);
  REQUIRE(RewardModulatedHebbian::eligibilityFunction(0.1, 1.0) == 0.1);
}

TEST_CASE("RewardModulatedHebbian - the high-pass signal uses the average before the row", "[RewardModulatedHebbian]") {
  Rule rule;
  rule.post = Rule::HIGH_PASS;
  const Eigen::RowVectorXd pre = row({1.0, 2.0});
  const Eigen::RowVectorXd v1 = row({0.5, -1.0, 2.0});
  const Eigen::RowVectorXd v2 = row({1.5, 0.0, 1.0});
  const Eigen::RowVectorXd v3 = row({-0.5, 3.0, 1.0});

  SECTION("a post_decay of 0 gives the first difference, starting from zero") {
    RewardModulatedHebbian engine = makeEngine(rule, 2, 3);
    REQUIRE_FALSE(engine.isPostAverageInitialized());
    commitRow(engine, pre, v1);
    REQUIRE(engine.getEligibility().isZero(0.0));
    REQUIRE(engine.getPostAverage() == v1);
    commitRow(engine, pre, v2);
    REQUIRE(engine.getEligibility().isApprox(pre.transpose() * (v2 - v1), 1e-15));
  }
  SECTION("a post_decay of 0.5 averages the potentials before the row") {
    rule.post_decay = 0.5;
    RewardModulatedHebbian engine = makeEngine(rule, 2, 3);
    commitRow(engine, pre, v1);
    commitRow(engine, pre, v2);
    REQUIRE(engine.getPostAverage().isApprox(0.5 * v1 + 0.5 * v2, 1e-15));
    commitRow(engine, pre, v3);
    REQUIRE(engine.getEligibility().isApprox(pre.transpose() * (v3 - (0.5 * v1 + 0.5 * v2)), 1e-15));
  }
  SECTION("resetActivity makes the next row start the average again") {
    RewardModulatedHebbian engine = makeEngine(rule, 2, 3);
    commitRow(engine, pre, v1);
    commitRow(engine, pre, v2);
    engine.resetActivity();
    REQUIRE_FALSE(engine.isPostAverageInitialized());
    commitRow(engine, pre, v3);
    REQUIRE(engine.getEligibility().isZero(0.0));
    REQUIRE(engine.getPostAverage() == v3);
  }
}

TEST_CASE("RewardModulatedHebbian - post signals and eligibility accumulation", "[RewardModulatedHebbian]") {
  const Eigen::RowVectorXd pre1 = row({1.0, -2.0});
  const Eigen::RowVectorXd pre2 = row({0.5, 3.0});
  const Eigen::RowVectorXd v1 = row({0.2, -0.4});
  const Eigen::RowVectorXd v2 = row({-1.0, 0.6});
  const Eigen::RowVectorXd xi1 = row({0.1, 0.0});
  const Eigen::RowVectorXd xi2 = row({-0.3, 0.2});
  Rule rule;

  SECTION("online noise eligibility decays by trace_decay") {
    rule.post = Rule::NOISE;
    rule.trace_decay = 0.5;
    RewardModulatedHebbian engine = makeEngine(rule, 2, 2);
    commitRow(engine, pre1, v1, xi1);
    commitRow(engine, pre2, v2, xi2);
    const Eigen::MatrixXd expected = 0.5 * pre1.transpose() * xi1 + pre2.transpose() * xi2;
    REQUIRE(engine.getEligibility().isApprox(expected, 1e-15));
  }
  SECTION("online eligibility with no decay keeps only the last row") {
    rule.post = Rule::POTENTIAL;
    RewardModulatedHebbian engine = makeEngine(rule, 2, 2);
    commitRow(engine, pre1, v1);
    commitRow(engine, pre2, v2);
    REQUIRE(engine.getEligibility() == Eigen::MatrixXd(pre2.transpose() * v2));
  }
  SECTION("trial eligibility and covariance accumulate, with S applied to both factors") {
    rule.post = Rule::POTENTIAL;
    rule.update = Rule::TRIAL;
    rule.eligibility_power = 3.0;
    RewardModulatedHebbian engine = makeEngine(rule, 2, 2);
    commitRow(engine, pre1, v1);
    commitRow(engine, pre2, v2);
    const Eigen::MatrixXd expected = pre1.array().cube().matrix().transpose() * v1.array().cube().matrix() +
                                     pre2.array().cube().matrix().transpose() * v2.array().cube().matrix();
    REQUIRE(engine.getEligibility().isApprox(expected, 1e-14));
    REQUIRE(engine.getCovariance().size() == 0);
  }
  SECTION("decorrelating trials accumulate the pre-synaptic covariance") {
    rule.post = Rule::NOISE;
    rule.update = Rule::TRIAL;
    rule.decorrelation = 1.0;
    RewardModulatedHebbian engine = makeEngine(rule, 2, 2);
    commitRow(engine, pre1, v1, xi1);
    commitRow(engine, pre2, v2, xi2);
    REQUIRE(engine.getCovariance().isApprox(pre1.transpose() * pre1 + pre2.transpose() * pre2, 1e-15));
    REQUIRE(engine.getCovarianceBound() == 4.0 + 9.0);
  }
}

TEST_CASE("RewardModulatedHebbian - reinforce needs an exploring row since the last update",
          "[RewardModulatedHebbian]") {
  Rule rule;
  rule.post = Rule::POTENTIAL;
  RewardModulatedHebbian engine = makeEngine(rule, 1, 1);
  REQUIRE_THROWS_AS(engine.checkReady(), std::logic_error);
  REQUIRE_THROWS_AS(engine.prepareReinforce(1.0, 0, std::nullopt), std::logic_error);
  commitRow(engine, row({1.0}), row({1.0}));
  reinforce(engine, 1.0);

  const Snapshot before(engine);
  REQUIRE_THROWS_AS(engine.prepareReinforce(2.0, 0, std::nullopt), std::logic_error);
  REQUIRE_THROWS_AS(engine.prepareModulate(1.0), std::logic_error);
  REQUIRE(Snapshot(engine) == before);
  REQUIRE_THROWS_AS(engine.commitUpdate(), std::logic_error);
  REQUIRE_THROWS_AS(engine.commitRow(), std::logic_error);
}

TEST_CASE("RewardModulatedHebbian - baselines", "[RewardModulatedHebbian]") {
  Rule rule;
  rule.post = Rule::POTENTIAL;
  rule.learning_rate = 0.5;
  const Eigen::RowVectorXd pre = row({2.0});
  const Eigen::RowVectorXd v = row({1.0}); // eligibility 2

  SECTION("EMA: the first reward of a key gives a zero modulator, then R - R_avg before the update") {
    rule.baseline = Rule::EMA;
    rule.baseline_decay = 0.75;
    RewardModulatedHebbian engine = makeEngine(rule, 1, 1);
    commitRow(engine, pre, v);
    engine.prepareReinforce(3.0, 0, std::nullopt);
    REQUIRE(engine.preparedDelta() == nullptr);
    engine.commitUpdate();
    REQUIRE(engine.getBaseline(0) == 3.0);
    commitRow(engine, pre, v);
    REQUIRE(reinforce(engine, 5.0)(0, 0) == 0.5 * (5.0 - 3.0) * 2.0);
    REQUIRE(engine.getBaseline(0) == 0.75 * 3.0 + 0.25 * 5.0);
    // Another key starts its own baseline.
    commitRow(engine, pre, v);
    REQUIRE(reinforce(engine, -4.0, 9)(0, 0) == 0.0);
    REQUIRE(engine.getBaseline(9) == -4.0);
    REQUIRE(engine.getBaseline(0) == 0.75 * 3.0 + 0.25 * 5.0);
  }
  SECTION("WINDOW: the mean of the last rewards, oldest dropped") {
    rule.baseline = Rule::WINDOW;
    rule.baseline_window = 2;
    RewardModulatedHebbian engine = makeEngine(rule, 1, 1);
    for (double reward : {1.0, 3.0, 8.0}) {
      commitRow(engine, pre, v);
      reinforce(engine, reward);
    }
    REQUIRE(engine.getBaselines().at(0) == std::vector<double>{3.0, 8.0});
    commitRow(engine, pre, v);
    REQUIRE(reinforce(engine, 10.0)(0, 0) == 0.5 * (10.0 - 5.5) * 2.0);
  }
  SECTION("NONE: the reward itself, with no state and no first-reward rule") {
    rule.baseline = Rule::NONE;
    RewardModulatedHebbian engine = makeEngine(rule, 1, 1);
    commitRow(engine, pre, v);
    REQUIRE(reinforce(engine, 3.0)(0, 0) == 0.5 * 3.0 * 2.0);
    REQUIRE(engine.getBaselines().empty());
    REQUIRE(engine.getBaseline(0) == 0.0);
  }
  SECTION("an explicit baseline is used as given and leaves the internal one alone") {
    rule.baseline = Rule::EMA;
    RewardModulatedHebbian engine = makeEngine(rule, 1, 1);
    commitRow(engine, pre, v);
    REQUIRE(reinforce(engine, 3.0, 0, 1.0)(0, 0) == 0.5 * (3.0 - 1.0) * 2.0);
    REQUIRE(engine.getBaselines().empty());
    commitRow(engine, pre, v);
    reinforce(engine, 2.0);
    commitRow(engine, pre, v);
    reinforce(engine, 7.0, 0, -1.0);
    REQUIRE(engine.getBaselines().at(0) == std::vector<double>{2.0});
  }
  SECTION("modulate leaves the baselines alone") {
    RewardModulatedHebbian engine = makeEngine(rule, 1, 1);
    commitRow(engine, pre, v);
    engine.prepareModulate(-2.0);
    REQUIRE((*engine.preparedDelta())(0, 0) == 0.5 * -2.0 * 2.0);
    engine.commitUpdate();
    REQUIRE(engine.getBaselines().empty());
  }
  SECTION("HEAVISIDE: 1 above the baseline, 0 at or below it") {
    rule.modulator = Rule::HEAVISIDE;
    rule.baseline = Rule::NONE;
    RewardModulatedHebbian engine = makeEngine(rule, 1, 1);
    commitRow(engine, pre, v);
    REQUIRE(reinforce(engine, 0.0)(0, 0) == 0.0);
    commitRow(engine, pre, v);
    REQUIRE(reinforce(engine, 1e-300)(0, 0) == 0.5 * 2.0);
    commitRow(engine, pre, v);
    REQUIRE(reinforce(engine, -3.0)(0, 0) == 0.0);
  }
}

TEST_CASE("RewardModulatedHebbian - learning rate schedule, clipping and update counting", "[RewardModulatedHebbian]") {
  Rule rule;
  rule.post = Rule::POTENTIAL;
  rule.baseline = Rule::NONE;
  rule.learning_rate = 1.0;
  rule.learning_rate_decay_steps = 2.0;
  RewardModulatedHebbian engine = makeEngine(rule, 1, 1);

  // eta_n = 1 / (1 + n / 2), n counting every successful update, zero modulators included.
  commitRow(engine, row({1.0}), row({1.0}));
  REQUIRE(reinforce(engine, 1.0)(0, 0) == 1.0);
  commitRow(engine, row({1.0}), row({1.0}));
  engine.prepareModulate(0.0);
  REQUIRE(engine.preparedDelta() == nullptr);
  engine.commitUpdate();
  REQUIRE(engine.getUpdateCount() == 2);
  // Rejected calls do not count.
  REQUIRE_THROWS_AS(engine.prepareReinforce(1.0, 0, std::nullopt), std::logic_error);
  commitRow(engine, row({1.0}), row({1.0}));
  REQUIRE_THROWS_AS(engine.prepareReinforce(nan_value, 0, std::nullopt), std::invalid_argument);
  REQUIRE(engine.getUpdateCount() == 2);
  REQUIRE(reinforce(engine, 1.0)(0, 0) == 0.5);

  rule.learning_rate_decay_steps.reset();
  rule.max_update = 0.25;
  RewardModulatedHebbian clipped = makeEngine(rule, 1, 2);
  commitRow(clipped, row({1.0}), row({0.1, -3.0}));
  const Eigen::MatrixXd delta = reinforce(clipped, 1.0);
  REQUIRE(delta(0, 0) == 0.1);
  REQUIRE(delta(0, 1) == -0.25);
}

TEST_CASE("RewardModulatedHebbian - updates clear trial eligibility and keep online eligibility",
          "[RewardModulatedHebbian]") {
  Rule rule;
  rule.post = Rule::NOISE;
  rule.baseline = Rule::NONE;
  const Eigen::RowVectorXd pre = row({1.0, 2.0});
  const Eigen::RowVectorXd xi = row({0.5});

  SECTION("online") {
    rule.trace_decay = 0.5;
    RewardModulatedHebbian engine = makeEngine(rule, 2, 1);
    commitRow(engine, pre, xi, xi);
    const Snapshot before(engine);
    reinforce(engine, 1.0);
    REQUIRE(engine.getEligibility() == before.eligibility);
    REQUIRE(engine.getEligibilityBound() == before.eligibility_bound);
    REQUIRE_FALSE(engine.hasEligibility());
  }
  SECTION("trial") {
    rule.update = Rule::TRIAL;
    rule.decorrelation = 1.0;
    RewardModulatedHebbian engine = makeEngine(rule, 2, 1);
    commitRow(engine, pre, xi, xi);
    reinforce(engine, 1.0);
    REQUIRE(engine.getEligibility().isZero(0.0));
    REQUIRE(engine.getCovariance().isZero(0.0));
    REQUIRE(engine.getEligibilityBound() == 0.0);
    REQUIRE(engine.getCovarianceBound() == 0.0);
  }
  SECTION("clearEligibility") {
    rule.update = Rule::TRIAL;
    rule.decorrelation = 1.0;
    RewardModulatedHebbian engine = makeEngine(rule, 2, 1);
    commitRow(engine, pre, xi, xi);
    engine.clearEligibility();
    REQUIRE(engine.getEligibility().isZero(0.0));
    REQUIRE(engine.getCovariance().isZero(0.0));
    REQUIRE(engine.getEligibilityBound() == 0.0);
    REQUIRE(engine.getCovarianceBound() == 0.0);
    REQUIRE_FALSE(engine.hasEligibility());
  }
}

TEST_CASE("RewardModulatedHebbian - decorrelated updates solve with the regularized covariance",
          "[RewardModulatedHebbian]") {
  Rule rule;
  rule.post = Rule::NOISE;
  rule.update = Rule::TRIAL;
  rule.baseline = Rule::NONE;
  rule.learning_rate = 0.3;
  rule.decorrelation = 0.5;
  RewardModulatedHebbian engine = makeEngine(rule, 3, 2);
  Eigen::MatrixXd C = Eigen::MatrixXd::Zero(3, 3);
  Eigen::MatrixXd E = Eigen::MatrixXd::Zero(3, 2);
  for (int t = 0; t < 4; ++t) {
    const Eigen::RowVectorXd pre = row({1.0 + t, 0.5 * t, -1.0});
    const Eigen::RowVectorXd xi = row({0.1 * t, -0.2});
    commitRow(engine, pre, xi, xi);
    C += pre.transpose() * pre;
    E += pre.transpose() * xi;
  }
  const Eigen::MatrixXd expected = 0.3 * 2.0 * (C + 0.5 * Eigen::MatrixXd::Identity(3, 3)).inverse() * E;
  REQUIRE(reinforce(engine, 2.0).isApprox(expected, 1e-12));
}

TEST_CASE("RewardModulatedHebbian - a rejected row changes no learning state", "[RewardModulatedHebbian]") {
  Rule rule;
  rule.post = Rule::NOISE;

  SECTION("non-finite potential, even with a finite noise post signal") {
    RewardModulatedHebbian engine = makeEngine(rule, 1, 1);
    commitRow(engine, row({1.0}), row({1.0}), row({0.5}));
    const Snapshot before(engine);
    REQUIRE_THROWS_AS(engine.stageRow(row({1.0}), row({inf_value}), row({0.5})), std::runtime_error);
    REQUIRE_THROWS_AS(engine.commitRow(), std::logic_error);
    REQUIRE(Snapshot(engine) == before);
  }
  SECTION("eligibility terms that overflow") {
    rule.eligibility_power = 3.0;
    RewardModulatedHebbian engine = makeEngine(rule, 1, 1);
    const Snapshot before(engine);
    REQUIRE_THROWS_AS(engine.stageRow(row({1e110}), row({1.0}), row({1.0})), std::runtime_error);
    REQUIRE(Snapshot(engine) == before);
  }
  SECTION("an outer product that overflows") {
    RewardModulatedHebbian engine = makeEngine(rule, 1, 1);
    const Snapshot before(engine);
    REQUIRE_THROWS_AS(engine.stageRow(row({1e200}), row({1.0}), row({1e200})), std::runtime_error);
    REQUIRE(Snapshot(engine) == before);
  }
  SECTION("trial accumulation is bounded without decay") {
    rule.update = Rule::TRIAL;
    RewardModulatedHebbian engine = makeEngine(rule, 1, 1);
    commitRow(engine, row({1e154}), row({1.0}), row({1e154}));
    REQUIRE(std::isfinite(engine.getEligibility()(0, 0)));
    const Snapshot before(engine);
    REQUIRE_THROWS_AS(engine.stageRow(row({1e154}), row({1.0}), row({1e154})), std::runtime_error);
    REQUIRE(Snapshot(engine) == before);
  }
  SECTION("the same rows are fine online, where the eligibility does not accumulate") {
    RewardModulatedHebbian engine = makeEngine(rule, 1, 1);
    commitRow(engine, row({1e154}), row({1.0}), row({1e154}));
    REQUIRE_NOTHROW(commitRow(engine, row({1e154}), row({1.0}), row({1e154})));
  }
  SECTION("a covariance that overflows although the eligibility does not") {
    rule.update = Rule::TRIAL;
    rule.decorrelation = 1.0;
    RewardModulatedHebbian engine = makeEngine(rule, 1, 1);
    const Snapshot before(engine);
    // S(pre) S(post) = 1e200 * 1e-200 = 1, but pre^2 = 1e400.
    REQUIRE_THROWS_AS(engine.stageRow(row({1e200}), row({1.0}), row({1e-200})), std::runtime_error);
    REQUIRE(Snapshot(engine) == before);
  }
}

TEST_CASE("RewardModulatedHebbian - a rejected update changes no learning state", "[RewardModulatedHebbian]") {
  Rule rule;
  rule.post = Rule::POTENTIAL;
  rule.baseline = Rule::EMA;

  SECTION("non-finite reward or baseline") {
    RewardModulatedHebbian engine = makeEngine(rule, 1, 1);
    commitRow(engine, row({1.0}), row({1.0}));
    reinforce(engine, 1.0);
    commitRow(engine, row({1.0}), row({1.0}));
    const Snapshot before(engine);
    REQUIRE_THROWS_AS(engine.prepareReinforce(nan_value, 0, std::nullopt), std::invalid_argument);
    REQUIRE_THROWS_AS(engine.prepareReinforce(inf_value, 0, std::nullopt), std::invalid_argument);
    REQUIRE_THROWS_AS(engine.prepareReinforce(1.0, 0, nan_value), std::invalid_argument);
    REQUIRE_THROWS_AS(engine.prepareModulate(inf_value), std::invalid_argument);
    REQUIRE(Snapshot(engine) == before);
  }
  SECTION("a reward difference that overflows") {
    RewardModulatedHebbian engine = makeEngine(rule, 1, 1);
    commitRow(engine, row({1.0}), row({1.0}));
    const Snapshot before(engine);
    REQUIRE_THROWS_AS(engine.prepareReinforce(1e308, 0, -1e308), std::runtime_error);
    REQUIRE(Snapshot(engine) == before);
  }
  SECTION("a weight change that overflows, also when clipping would hide it") {
    rule.learning_rate = 1e300;
    rule.max_update = 1.0;
    rule.baseline = Rule::NONE;
    RewardModulatedHebbian engine = makeEngine(rule, 1, 1);
    commitRow(engine, row({1e10}), row({1.0}));
    const Snapshot before(engine);
    REQUIRE_THROWS_AS(engine.prepareReinforce(1.0, 0, std::nullopt), std::runtime_error);
    REQUIRE(Snapshot(engine) == before);
  }
  SECTION("a decorrelation solve that overflows") {
    rule.post = Rule::NOISE;
    rule.update = Rule::TRIAL;
    rule.baseline = Rule::NONE;
    rule.decorrelation = 1e-300;
    RewardModulatedHebbian engine = makeEngine(rule, 1, 1);
    // pre^2 underflows to zero, so the solve divides the eligibility 1e-200 * 1e250 by 1e-300.
    commitRow(engine, row({1e-200}), row({1.0}), row({1e250}));
    REQUIRE(engine.getCovariance()(0, 0) == 0.0);
    const Snapshot before(engine);
    REQUIRE_THROWS_AS(engine.prepareReinforce(1.0, 0, std::nullopt), std::runtime_error);
    REQUIRE(Snapshot(engine) == before);
  }
}

TEST_CASE("RewardModulatedHebbian - baselines of finite rewards stay finite", "[RewardModulatedHebbian]") {
  Rule rule;
  rule.post = Rule::POTENTIAL;
  constexpr double huge = std::numeric_limits<double>::max();
  SECTION("a window of the largest rewards") {
    rule.baseline = Rule::WINDOW;
    rule.baseline_window = 3;
  }
  SECTION("an exponential average of the largest rewards") {
    rule.baseline = Rule::EMA;
    rule.baseline_decay = GENERATE(0.0, 0.3, 0.8, 0.999);
  }
  RewardModulatedHebbian engine = makeEngine(rule, 1, 1);
  for (int i = 0; i < 5; ++i) {
    commitRow(engine, row({1.0}), row({1.0}));
    REQUIRE_NOTHROW(reinforce(engine, huge));
    REQUIRE(std::isfinite(*engine.getBaseline(0)));
  }
  REQUIRE(*engine.getBaseline(0) == huge);
  // The baseline equals the rewards, so the next difference is zero, not an overflow.
  commitRow(engine, row({1.0}), row({1.0}));
  engine.prepareReinforce(huge, 0, std::nullopt);
  REQUIRE(engine.preparedDelta() == nullptr);
}

TEST_CASE("RewardModulatedHebbian - the window baseline is the mean of its rewards", "[RewardModulatedHebbian]") {
  Rule rule;
  rule.post = Rule::POTENTIAL;
  rule.baseline = Rule::WINDOW;
  rule.baseline_window = 4;
  RewardModulatedHebbian engine = makeEngine(rule, 1, 1);
  for (double reward : {-2.0, 6.0, 0.5, 3.5, 10.0}) {
    commitRow(engine, row({1.0}), row({1.0}));
    reinforce(engine, reward);
  }
  REQUIRE_THAT(*engine.getBaseline(0), Catch::Matchers::WithinRel(5.0, 1e-15)); // 6, 0.5, 3.5, 10
}
