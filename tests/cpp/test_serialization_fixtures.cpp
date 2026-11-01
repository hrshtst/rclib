// Loads the frozen fixtures of model format versions 1 to 4 (tests/data/serialization/v<N>,
// see generate.py there) to check that files written by earlier builds keep
// loading with the same configuration and results.

#include "rclib/Model.h"
#include "rclib/Serialization.h"
#include "rclib/readouts/LmsReadout.h"
#include "rclib/readouts/RewardModulatedReadout.h"
#include "rclib/readouts/RidgeReadout.h"
#include "rclib/readouts/RlsReadout.h"
#include "rclib/reservoirs/NvarReservoir.h"
#include "rclib/reservoirs/RandomSparseReservoir.h"
#include "rclib/reservoirs/RewardModulatedReservoir.h"

#include <Eigen/Dense>
#include <catch2/catch_all.hpp>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::filesystem::path fixtureDirectory(int version) {
  return std::filesystem::path(RCLIB_TEST_DATA_DIR) / "serialization" / ("v" + std::to_string(version));
}

std::string fixturePath(int version, const std::string &name) {
  return (fixtureDirectory(version) / (name + ".rclib")).string();
}

// Parses "<name> <values...>" lines into one-column arrays, skipping '#' comments.
std::map<std::string, Eigen::MatrixXd> readExpected(int version, const std::string &name) {
  std::ifstream file(fixtureDirectory(version) / (name + ".expected.txt"));
  REQUIRE(file);
  std::map<std::string, Eigen::MatrixXd> arrays;
  std::string line;
  while (std::getline(file, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }
    std::istringstream fields(line);
    std::string key;
    fields >> key;
    std::vector<double> values;
    double value = 0.0;
    while (fields >> value) {
      values.push_back(value);
    }
    REQUIRE(fields.eof()); // every field parsed as a number
    arrays[key] = Eigen::Map<const Eigen::MatrixXd>(values.data(), static_cast<Eigen::Index>(values.size()), 1);
  }
  return arrays;
}

// The fixtures are small and numerically stable, so results computed by another
// build agree to a tight tolerance. This is not a general portability bound.
void requireClose(const Eigen::MatrixXd &actual, const Eigen::MatrixXd &expected) {
  REQUIRE(actual.rows() == expected.rows());
  REQUIRE(actual.cols() == expected.cols());
  for (Eigen::Index i = 0; i < expected.size(); ++i) {
    CHECK_THAT(actual(i),
               Catch::Matchers::WithinRel(expected(i), 1e-10) || Catch::Matchers::WithinAbs(expected(i), 1e-12));
  }
}

// Loads a fixture. Re-saving a fixture of the current format version must reproduce
// the file byte for byte, which pins the writer to that version as well as the reader.
// The writer only writes the current version, so an older fixture is re-saved and
// reloaded instead; the checks that follow then show that its configuration and
// results survive the conversion.
Model loadFixture(int version, const std::string &name) {
  Model model = Model::load(fixturePath(version, name));
  std::stringstream saved;
  model.save(saved);
  if (version != static_cast<int>(serialization_format_version)) {
    return Model::load(saved);
  }
  std::ifstream file(fixturePath(version, name), std::ios::binary);
  std::ostringstream fixture_bytes;
  fixture_bytes << file.rdbuf();
  REQUIRE(saved.str() == fixture_bytes.str());
  return model;
}

template <typename T, typename Base> std::shared_ptr<T> as(const std::shared_ptr<Base> &component) {
  auto concrete = std::dynamic_pointer_cast<T>(component);
  REQUIRE(concrete);
  return concrete;
}

// The expected results follow the sequence documented in generate.py.
void requireExpectedResults(Model &model, int version, const std::string &name, bool online_readout) {
  const auto expected = readExpected(version, name);
  requireClose(model.predictOnline(expected.at("input_online")), expected.at("online"));
  requireClose(model.predict(expected.at("input_predict")), expected.at("predict"));
  if (online_readout) {
    model.partialFit(expected.at("input_partial_fit"), expected.at("target_partial_fit"));
    requireClose(model.predict(expected.at("input_predict")), expected.at("after_partial_fit"));
  }
}

} // namespace

TEST_CASE("Serialization fixtures - serial RandomSparse -> NVAR with Ridge", "[serialization][fixtures]") {
  const int version = GENERATE(1, 2, 3, 4);
  CAPTURE(version);
  Model model = loadFixture(version, "serial_rs_nvar_ridge");
  REQUIRE(model.getConnectionType() == "serial");
  REQUIRE(model.getNumReservoirs() == 2);

  const auto random_sparse = as<RandomSparseReservoir>(model.getReservoir(0));
  REQUIRE(random_sparse->getNNeurons() == 8);
  REQUIRE(random_sparse->getSpectralRadius() == 0.9);
  REQUIRE(random_sparse->getSparsity() == 0.5);
  REQUIRE(random_sparse->getLeakRate() == 0.5);
  REQUIRE(random_sparse->getInputScaling() == 1.0);
  REQUIRE(random_sparse->getIncludeBias());
  REQUIRE(random_sparse->getSeed() == 1U);
  REQUIRE(random_sparse->getSpectralRadiusMethod() == RandomSparseReservoir::POWER_ITERATION);
  REQUIRE(random_sparse->getBiasScaling() == 1.0);
  REQUIRE(random_sparse->getInputDim() == 1);
  const auto nvar = as<NvarReservoir>(model.getReservoir(1));
  REQUIRE(nvar->getNumLags() == 2);
  REQUIRE(nvar->getPolynomialOrder() == 2);
  REQUIRE(nvar->getInputDim() == 8);
  const auto ridge = as<RidgeReadout>(model.getReadout());
  REQUIRE(ridge->getAlpha() == 1e-3);
  REQUIRE(ridge->getIncludeBias());
  REQUIRE(ridge->getSolver() == RidgeReadout::AUTO);
  REQUIRE(ridge->getEffectiveSolver() == RidgeReadout::DUAL_CHOLESKY);
  REQUIRE(ridge->getTolerance() == 1e-10);
  REQUIRE(ridge->getInputDim() == 152);

  requireExpectedResults(model, version, "serial_rs_nvar_ridge", false);
}

TEST_CASE("Serialization fixtures - parallel RandomSparse + NVAR with rank-k RLS", "[serialization][fixtures]") {
  const int version = GENERATE(1, 2, 3, 4);
  CAPTURE(version);
  Model model = loadFixture(version, "parallel_rs_nvar_rls");
  REQUIRE(model.getConnectionType() == "parallel");
  REQUIRE(model.getNumReservoirs() == 2);

  const auto random_sparse = as<RandomSparseReservoir>(model.getReservoir(0));
  REQUIRE(random_sparse->getNNeurons() == 8);
  REQUIRE(random_sparse->getSpectralRadius() == 0.8);
  REQUIRE(random_sparse->getSparsity() == 0.4);
  REQUIRE(random_sparse->getLeakRate() == 0.7);
  REQUIRE(random_sparse->getInputScaling() == 0.5);
  REQUIRE_FALSE(random_sparse->getIncludeBias());
  REQUIRE(random_sparse->getSeed() == 2U);
  REQUIRE(random_sparse->getSpectralRadiusMethod() == RandomSparseReservoir::POWER_ITERATION);
  REQUIRE(random_sparse->getBiasScaling() == 1.0);
  const auto nvar = as<NvarReservoir>(model.getReservoir(1));
  REQUIRE(nvar->getNumLags() == 2);
  REQUIRE(nvar->getPolynomialOrder() == 1);
  const auto rls = as<RlsReadout>(model.getReadout());
  REQUIRE(rls->getLambda() == 1.0);
  REQUIRE(rls->getDelta() == 0.5);
  REQUIRE(rls->getIncludeBias());
  REQUIRE(rls->getSolver() == RlsReadout::RANK_K_UPDATE);
  REQUIRE(rls->getInputDim() == 10);

  requireExpectedResults(model, version, "parallel_rs_nvar_rls", true);
}

TEST_CASE("Serialization fixtures - serial RandomSparse with LMS", "[serialization][fixtures]") {
  const int version = GENERATE(1, 2, 3, 4);
  CAPTURE(version);
  Model model = loadFixture(version, "serial_rs_lms");
  REQUIRE(model.getConnectionType() == "serial");
  REQUIRE(model.getNumReservoirs() == 1);

  const auto random_sparse = as<RandomSparseReservoir>(model.getReservoir(0));
  REQUIRE(random_sparse->getNNeurons() == 8);
  REQUIRE(random_sparse->getSeed() == 3U);
  REQUIRE(random_sparse->getSpectralRadiusMethod() == RandomSparseReservoir::POWER_ITERATION);
  REQUIRE(random_sparse->getBiasScaling() == 1.0);
  const auto lms = as<LmsReadout>(model.getReadout());
  REQUIRE(lms->getLearningRate() == 0.05);
  REQUIRE(lms->getIncludeBias());
  REQUIRE(lms->getInputDim() == 8);

  requireExpectedResults(model, version, "serial_rs_lms", true);
}

TEST_CASE("Serialization fixtures - uninitialized reservoir and unfitted Ridge", "[serialization][fixtures]") {
  const int version = GENERATE(1, 2, 3, 4);
  CAPTURE(version);
  Model model = loadFixture(version, "unfitted_ridge");
  REQUIRE(model.getNumReservoirs() == 1);

  const auto random_sparse = as<RandomSparseReservoir>(model.getReservoir(0));
  REQUIRE(random_sparse->getNNeurons() == 6);
  REQUIRE(random_sparse->getSeed() == 4U);
  REQUIRE(random_sparse->getSpectralRadiusMethod() == RandomSparseReservoir::POWER_ITERATION);
  REQUIRE(random_sparse->getBiasScaling() == 1.0);
  REQUIRE(random_sparse->getInputDim() == 0);
  const auto ridge = as<RidgeReadout>(model.getReadout());
  REQUIRE(ridge->getAlpha() == 0.5);
  REQUIRE_FALSE(ridge->getIncludeBias());
  REQUIRE(ridge->getSolver() == RidgeReadout::CHOLESKY);
  REQUIRE(ridge->getInputDim() == 0);

  REQUIRE_THROWS_WITH(model.predict(Eigen::MatrixXd::Ones(3, 1)), Catch::Matchers::ContainsSubstring("must be fit"));
}

TEST_CASE("Serialization fixtures v2+ - serial RandomSparse scaled by the dense method with Ridge",
          "[serialization][fixtures]") {
  const int version = GENERATE(2, 3, 4);
  CAPTURE(version);
  Model model = loadFixture(version, "serial_rs_dense_ridge");
  REQUIRE(model.getConnectionType() == "serial");
  REQUIRE(model.getNumReservoirs() == 1);

  const auto random_sparse = as<RandomSparseReservoir>(model.getReservoir(0));
  REQUIRE(random_sparse->getNNeurons() == 8);
  REQUIRE(random_sparse->getSpectralRadius() == 0.9);
  REQUIRE(random_sparse->getSparsity() == 0.5);
  REQUIRE(random_sparse->getLeakRate() == 0.5);
  REQUIRE(random_sparse->getInputScaling() == 1.0);
  REQUIRE(random_sparse->getIncludeBias());
  REQUIRE(random_sparse->getSeed() == 5U);
  REQUIRE(random_sparse->getSpectralRadiusMethod() == RandomSparseReservoir::DENSE);
  REQUIRE(random_sparse->getBiasScaling() == 1.0);
  REQUIRE(random_sparse->getInputDim() == 1);
  const auto ridge = as<RidgeReadout>(model.getReadout());
  REQUIRE(ridge->getAlpha() == 1e-3);
  REQUIRE(ridge->getIncludeBias());
  REQUIRE(ridge->getSolver() == RidgeReadout::AUTO);
  REQUIRE(ridge->getEffectiveSolver() == RidgeReadout::CHOLESKY);
  REQUIRE(ridge->getInputDim() == 8);

  requireExpectedResults(model, version, "serial_rs_dense_ridge", false);
}

TEST_CASE("Serialization fixtures v3+ - serial RandomSparse with a scaled bias with Ridge",
          "[serialization][fixtures]") {
  const int version = GENERATE(3, 4);
  CAPTURE(version);
  Model model = loadFixture(version, "serial_rs_bias_ridge");
  REQUIRE(model.getConnectionType() == "serial");
  REQUIRE(model.getNumReservoirs() == 1);

  const auto random_sparse = as<RandomSparseReservoir>(model.getReservoir(0));
  REQUIRE(random_sparse->getNNeurons() == 8);
  REQUIRE(random_sparse->getSpectralRadius() == 0.9);
  REQUIRE(random_sparse->getSparsity() == 0.5);
  REQUIRE(random_sparse->getLeakRate() == 0.5);
  REQUIRE(random_sparse->getInputScaling() == 1.0);
  REQUIRE(random_sparse->getIncludeBias());
  REQUIRE(random_sparse->getSeed() == 6U);
  REQUIRE(random_sparse->getSpectralRadiusMethod() == RandomSparseReservoir::POWER_ITERATION);
  REQUIRE(random_sparse->getBiasScaling() == 0.25);
  REQUIRE(random_sparse->getBias().cwiseAbs().maxCoeff() <= 0.25);
  REQUIRE(random_sparse->getInputDim() == 1);
  const auto ridge = as<RidgeReadout>(model.getReadout());
  REQUIRE(ridge->getAlpha() == 1e-3);
  REQUIRE(ridge->getIncludeBias());
  REQUIRE(ridge->getSolver() == RidgeReadout::AUTO);
  REQUIRE(ridge->getEffectiveSolver() == RidgeReadout::CHOLESKY);
  REQUIRE(ridge->getInputDim() == 8);

  requireExpectedResults(model, version, "serial_rs_bias_ridge", false);
}

namespace {

using Rule = RewardModulatedHebbianRule;

// The reward fixtures' sequence documented in v4/generate.py: explore, reward, noise-free steps.
void requireRewardResults(Model &model, const std::string &name, Eigen::Index input_width) {
  const auto expected = readExpected(4, name);
  const auto rows = [&](const std::string &key) {
    const Eigen::MatrixXd &column = expected.at(key);
    // Stored row by row, as numpy ravels a 2-D array.
    Eigen::MatrixXd values(column.rows() / input_width, input_width);
    for (Eigen::Index i = 0; i < column.rows(); ++i) {
      values(i / input_width, i % input_width) = column(i);
    }
    return values;
  };
  const auto flatten = [](const Eigen::MatrixXd &matrix) {
    Eigen::MatrixXd column(matrix.size(), 1);
    for (Eigen::Index i = 0; i < matrix.size(); ++i) {
      column(i) = matrix(i / matrix.cols(), i % matrix.cols());
    }
    return column;
  };
  requireClose(flatten(model.act(rows("input_act"))), expected.at("explore"));
  model.reinforce(expected.at("reward")(0), static_cast<int>(expected.at("key")(0)));
  requireClose(flatten(model.act(rows("input_noise_free"), false)), expected.at("noise_free"));
}

} // namespace

TEST_CASE("Serialization fixtures v4 - reward-modulated components saved mid-learning", "[serialization][fixtures]") {
  SECTION("serial RandomSparse with an online Heaviside readout") {
    Model model = loadFixture(4, "reward_serial_rs_readout");
    REQUIRE(model.getConnectionType() == "serial");
    REQUIRE(as<RandomSparseReservoir>(model.getReservoir(0))->getSeed() == 7U);
    const auto readout = as<RewardModulatedReadout>(model.getReadout());
    REQUIRE(readout->getNOutputs() == 2);
    REQUIRE(readout->getIncludeBias());
    const Rule &rule = readout->getRule();
    REQUIRE(rule.modulator == Rule::HEAVISIDE);
    REQUIRE(rule.post == Rule::HIGH_PASS);
    REQUIRE(rule.learning_rate_decay_steps == 20.0);
    REQUIRE(rule.noise_seed == 11U);
    REQUIRE(readout->getLearning().hasEligibility());
    REQUIRE(readout->getLearning().getUpdateCount() == 4);
    REQUIRE(readout->getLearning().getBaselines().size() == 2);
    requireRewardResults(model, "reward_serial_rs_readout", 1);
  }
  SECTION("a reward-modulated reservoir learning by trials, with a Ridge readout") {
    Model model = loadFixture(4, "reward_reservoir_ridge");
    const auto reservoir = as<RewardModulatedReservoir>(model.getReservoir(0));
    REQUIRE(reservoir->getNNeurons() == 8);
    REQUIRE(reservoir->getLeakRate() == 0.4);
    REQUIRE(reservoir->getPlasticPresynaptic() == std::vector<int>{0, 1, 2, 3, 4, 5});
    REQUIRE(reservoir->getPlasticPostsynaptic() == std::vector<int>{1, 2, 3, 5, 6, 7});
    const Rule &rule = reservoir->getRule();
    REQUIRE(rule.update == Rule::TRIAL);
    REQUIRE(rule.eligibility_power == 3.0);
    REQUIRE(rule.baseline == Rule::WINDOW);
    REQUIRE(rule.baseline_window == 2);
    REQUIRE(reservoir->getLearning().getBaselines().size() == 3);
    REQUIRE(reservoir->getLearning().hasEligibility());
    as<RidgeReadout>(model.getReadout());
    requireRewardResults(model, "reward_reservoir_ridge", 1);
  }
  SECTION("an identity NVAR with a decorrelated trial readout") {
    Model model = loadFixture(4, "reward_nvar_decorrelated");
    const auto readout = as<RewardModulatedReadout>(model.getReadout());
    REQUIRE(readout->getRule().decorrelation == 1.0);
    REQUIRE(readout->getRule().noise_distribution == Rule::GAUSSIAN);
    REQUIRE(readout->getLearning().getCovariance().rows() == 3);
    requireRewardResults(model, "reward_nvar_decorrelated", 3);
  }
  SECTION("a reward-modulated reservoir without a readout") {
    Model model = loadFixture(4, "reward_reservoir_no_readout");
    REQUIRE_FALSE(model.hasReadout());
    const auto reservoir = as<RewardModulatedReservoir>(model.getReservoir(0));
    REQUIRE(reservoir->getRule().post_decay == 0.8);
    REQUIRE(reservoir->getLearning().isPostAverageInitialized());
    requireRewardResults(model, "reward_reservoir_no_readout", 2);
  }
}
