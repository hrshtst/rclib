// Loads the frozen fixtures of model format version 1 (tests/data/serialization/v1,
// see generate.py there) to check that files written by earlier builds keep
// loading with the same configuration and results.

#include "rclib/Model.h"
#include "rclib/Serialization.h"
#include "rclib/readouts/LmsReadout.h"
#include "rclib/readouts/RidgeReadout.h"
#include "rclib/readouts/RlsReadout.h"
#include "rclib/reservoirs/NvarReservoir.h"
#include "rclib/reservoirs/RandomSparseReservoir.h"

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

const std::filesystem::path fixture_directory = std::filesystem::path(RCLIB_TEST_DATA_DIR) / "serialization" / "v1";

std::string fixturePath(const std::string &name) { return (fixture_directory / (name + ".rclib")).string(); }

// Parses "<name> <values...>" lines into one-column arrays, skipping '#' comments.
std::map<std::string, Eigen::MatrixXd> readExpected(const std::string &name) {
  std::ifstream file(fixture_directory / (name + ".expected.txt"));
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

// Re-saving a loaded fixture must reproduce the file byte for byte, which pins the
// writer to format version 1 as well as the reader.
void requireResavesIdentically(const Model &model, const std::string &name) {
  std::ifstream file(fixturePath(name), std::ios::binary);
  std::ostringstream fixture_bytes;
  fixture_bytes << file.rdbuf();
  std::ostringstream saved;
  model.save(saved);
  REQUIRE(saved.str() == fixture_bytes.str());
}

template <typename T, typename Base> std::shared_ptr<T> as(const std::shared_ptr<Base> &component) {
  auto concrete = std::dynamic_pointer_cast<T>(component);
  REQUIRE(concrete);
  return concrete;
}

// The expected results follow the sequence documented in generate.py.
void requireExpectedResults(Model &model, const std::string &name, bool online_readout) {
  const auto expected = readExpected(name);
  requireClose(model.predictOnline(expected.at("input_online")), expected.at("online"));
  requireClose(model.predict(expected.at("input_predict")), expected.at("predict"));
  if (online_readout) {
    model.partialFit(expected.at("input_partial_fit"), expected.at("target_partial_fit"));
    requireClose(model.predict(expected.at("input_predict")), expected.at("after_partial_fit"));
  }
}

} // namespace

TEST_CASE("Serialization fixtures v1 - serial RandomSparse -> NVAR with Ridge", "[serialization][fixtures]") {
  Model model = Model::load(fixturePath("serial_rs_nvar_ridge"));
  requireResavesIdentically(model, "serial_rs_nvar_ridge");
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

  requireExpectedResults(model, "serial_rs_nvar_ridge", false);
}

TEST_CASE("Serialization fixtures v1 - parallel RandomSparse + NVAR with rank-k RLS", "[serialization][fixtures]") {
  Model model = Model::load(fixturePath("parallel_rs_nvar_rls"));
  requireResavesIdentically(model, "parallel_rs_nvar_rls");
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
  const auto nvar = as<NvarReservoir>(model.getReservoir(1));
  REQUIRE(nvar->getNumLags() == 2);
  REQUIRE(nvar->getPolynomialOrder() == 1);
  const auto rls = as<RlsReadout>(model.getReadout());
  REQUIRE(rls->getLambda() == 1.0);
  REQUIRE(rls->getDelta() == 0.5);
  REQUIRE(rls->getIncludeBias());
  REQUIRE(rls->getSolver() == RlsReadout::RANK_K_UPDATE);
  REQUIRE(rls->getInputDim() == 10);

  requireExpectedResults(model, "parallel_rs_nvar_rls", true);
}

TEST_CASE("Serialization fixtures v1 - serial RandomSparse with LMS", "[serialization][fixtures]") {
  Model model = Model::load(fixturePath("serial_rs_lms"));
  requireResavesIdentically(model, "serial_rs_lms");
  REQUIRE(model.getConnectionType() == "serial");
  REQUIRE(model.getNumReservoirs() == 1);

  const auto random_sparse = as<RandomSparseReservoir>(model.getReservoir(0));
  REQUIRE(random_sparse->getNNeurons() == 8);
  REQUIRE(random_sparse->getSeed() == 3U);
  const auto lms = as<LmsReadout>(model.getReadout());
  REQUIRE(lms->getLearningRate() == 0.05);
  REQUIRE(lms->getIncludeBias());
  REQUIRE(lms->getInputDim() == 8);

  requireExpectedResults(model, "serial_rs_lms", true);
}

TEST_CASE("Serialization fixtures v1 - uninitialized reservoir and unfitted Ridge", "[serialization][fixtures]") {
  Model model = Model::load(fixturePath("unfitted_ridge"));
  requireResavesIdentically(model, "unfitted_ridge");
  REQUIRE(model.getNumReservoirs() == 1);

  const auto random_sparse = as<RandomSparseReservoir>(model.getReservoir(0));
  REQUIRE(random_sparse->getNNeurons() == 6);
  REQUIRE(random_sparse->getSeed() == 4U);
  REQUIRE(random_sparse->getInputDim() == 0);
  const auto ridge = as<RidgeReadout>(model.getReadout());
  REQUIRE(ridge->getAlpha() == 0.5);
  REQUIRE_FALSE(ridge->getIncludeBias());
  REQUIRE(ridge->getSolver() == RidgeReadout::CHOLESKY);
  REQUIRE(ridge->getInputDim() == 0);

  REQUIRE_THROWS_WITH(model.predict(Eigen::MatrixXd::Ones(3, 1)), Catch::Matchers::ContainsSubstring("must be fit"));
}
