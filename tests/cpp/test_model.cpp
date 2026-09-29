#include "rclib/Model.h"
#include "rclib/Serialization.h"
#include "rclib/readouts/RidgeReadout.h"
#include "rclib/readouts/RlsReadout.h"
#include "rclib/reservoirs/NvarReservoir.h"
#include "rclib/reservoirs/RandomSparseReservoir.h"

#include <Eigen/Dense>
#include <catch2/catch_all.hpp>
#include <cmath>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>

TEST_CASE("Model - configuration", "[Model]") {
  Model model;

  SECTION("Throws when not configured") {
    Eigen::MatrixXd inputs = Eigen::MatrixXd::Random(10, 5);
    Eigen::MatrixXd targets = Eigen::MatrixXd::Random(10, 2);
    REQUIRE_THROWS(model.fit(inputs, targets));
    REQUIRE_THROWS(model.predict(inputs));
    REQUIRE_THROWS(model.predictOnline(inputs.row(0)));
  }

  auto res = std::make_shared<RandomSparseReservoir>(10, 0.9, 0.5, 0.1, 1.0);
  model.addReservoir(res);

  SECTION("Throws when readout is not set") {
    Eigen::MatrixXd inputs = Eigen::MatrixXd::Random(10, 5);
    Eigen::MatrixXd targets = Eigen::MatrixXd::Random(10, 2);
    REQUIRE_THROWS(model.fit(inputs, targets));
    REQUIRE_THROWS(model.predict(inputs));
    REQUIRE_THROWS(model.predictOnline(inputs.row(0)));
  }

  auto readout = std::make_shared<RidgeReadout>();
  model.setReadout(readout);

  SECTION("Does not throw when fully configured") {
    Eigen::MatrixXd inputs = Eigen::MatrixXd::Random(10, 5);
    Eigen::MatrixXd targets = Eigen::MatrixXd::Random(10, 2);
    REQUIRE_NOTHROW(model.fit(inputs, targets));
    REQUIRE_NOTHROW(model.predict(inputs));
    REQUIRE_NOTHROW(model.predictOnline(inputs.row(0)));
  }
}

TEST_CASE("Model - fit and predict", "[Model]") {
  Model model;
  auto res = std::make_shared<RandomSparseReservoir>(100, 0.9, 0.1, 0.2, 1.0);
  auto readout = std::make_shared<RidgeReadout>(1e-6);
  model.addReservoir(res);
  model.setReadout(readout);

  Eigen::MatrixXd inputs = Eigen::MatrixXd::Random(200, 1);
  Eigen::MatrixXd targets = Eigen::MatrixXd::Random(200, 1);

  model.fit(inputs, targets);
  Eigen::MatrixXd predictions = model.predict(inputs);

  REQUIRE(predictions.rows() == 200);
  REQUIRE(predictions.cols() == 1);

  double prediction_error = (predictions - targets).squaredNorm();
  double original_error = targets.squaredNorm();
  REQUIRE(prediction_error < original_error);
}

TEST_CASE("Model - parallel connection", "[Model]") {
  Model model;
  auto res1 = std::make_shared<RandomSparseReservoir>(50, 0.9, 0.1, 0.2, 1.0);
  auto res2 = std::make_shared<RandomSparseReservoir>(50, 0.9, 0.1, 0.2, 1.0);
  auto readout = std::make_shared<RidgeReadout>(1e-6);
  model.addReservoir(res1, "parallel");
  model.addReservoir(res2, "parallel");
  model.setReadout(readout);

  Eigen::MatrixXd inputs = Eigen::MatrixXd::Random(200, 1);
  Eigen::MatrixXd targets = Eigen::MatrixXd::Random(200, 1);

  model.fit(inputs, targets);
  Eigen::MatrixXd predictions = model.predict(inputs);

  REQUIRE(predictions.rows() == 200);
  REQUIRE(predictions.cols() == 1);

  double prediction_error = (predictions - targets).squaredNorm();
  double original_error = targets.squaredNorm();
  REQUIRE(prediction_error < original_error);
}

TEST_CASE("Model - parallel reservoir errors propagate to the caller", "[Model]") {
  // Reservoir updates in a parallel model run inside an OpenMP region; an error
  // raised there must reach the caller as an exception instead of terminating.
  // Both reservoir types lock their input width on first use.
  Model model;
  SECTION("NvarReservoir") {
    model.addReservoir(std::make_shared<NvarReservoir>(2), "parallel");
    model.addReservoir(std::make_shared<NvarReservoir>(3), "parallel");
  }
  SECTION("RandomSparseReservoir") {
    model.addReservoir(std::make_shared<RandomSparseReservoir>(10, 0.9, 0.5, 0.1, 1.0), "parallel");
    model.addReservoir(std::make_shared<RandomSparseReservoir>(8, 0.9, 0.5, 0.1, 1.0), "parallel");
  }
  model.setReadout(std::make_shared<RidgeReadout>(1e-6));

  Eigen::MatrixXd inputs = Eigen::MatrixXd::Random(20, 1);
  model.fit(inputs, inputs);

  Eigen::MatrixXd wider = Eigen::MatrixXd::Random(5, 2);
  REQUIRE_THROWS_AS(model.predict(wider), std::invalid_argument);
  REQUIRE_THROWS_AS(model.predictOnline(wider.row(0)), std::invalid_argument);

  // The model remains usable with the original input width.
  REQUIRE(model.predict(inputs).rows() == inputs.rows());
}

TEST_CASE("Model - resetReservoirs", "[Model]") {
  Model model;
  auto res1 = std::make_shared<RandomSparseReservoir>(10, 0.9, 0.1, 0.2, 1.0);
  auto res2 = std::make_shared<RandomSparseReservoir>(5, 0.8, 0.2, 0.3, 1.0);
  model.addReservoir(res1);
  model.addReservoir(res2);

  // Advance states to ensure they are not zero
  Eigen::MatrixXd input = Eigen::MatrixXd::Ones(1, 1);
  res1->advance(input);
  res2->advance(input);

  REQUIRE(res1->getState().norm() > 0);
  REQUIRE(res2->getState().norm() > 0);

  model.resetReservoirs();

  REQUIRE(res1->getState().norm() == 0);
  REQUIRE(res2->getState().norm() == 0);
}

TEST_CASE("Model - partialFit", "[Model]") {
  Model model;
  auto res = std::make_shared<RandomSparseReservoir>(100, 0.9, 0.1, 0.2, 1.0);
  model.addReservoir(res);

  SECTION("Throws when readout not set") {
    REQUIRE_THROWS(model.partialFit(Eigen::MatrixXd::Random(1, 1), Eigen::MatrixXd::Random(1, 1)));
  }

  // Use RLS for online updates
  auto readout = std::make_shared<RlsReadout>(0.99, 1.0, true);
  model.setReadout(readout);

  SECTION("Updates weights on partialFit") {
    Eigen::MatrixXd input = Eigen::MatrixXd::Random(1, 1);
    Eigen::MatrixXd target = Eigen::MatrixXd::Random(1, 1);

    // Initialize with a dummy fit to ensure W_out is allocated
    model.partialFit(input, target);

    // Get prediction after initial fit
    Eigen::MatrixXd pred_before = model.predict(input, true);

    // Perform another partial fit
    model.partialFit(input, target);

    // Prediction should have changed
    Eigen::MatrixXd pred_after = model.predict(input, true);
    REQUIRE(pred_before(0, 0) != Catch::Approx(pred_after(0, 0)));
  }

  SECTION("Parallel connection in partialFit") {
    Model pmodel;
    auto res1 = std::make_shared<RandomSparseReservoir>(50, 0.9, 0.1, 0.2, 1.0);
    auto res2 = std::make_shared<RandomSparseReservoir>(50, 0.9, 0.1, 0.2, 1.0);
    pmodel.addReservoir(res1, "parallel");
    pmodel.addReservoir(res2, "parallel");
    pmodel.setReadout(std::make_shared<RlsReadout>());

    Eigen::MatrixXd input = Eigen::MatrixXd::Random(1, 1);
    Eigen::MatrixXd target = Eigen::MatrixXd::Random(1, 1);

    REQUIRE_NOTHROW(pmodel.partialFit(input, target));
  }
}

TEST_CASE("Model - reservoir count and connection type", "[Model]") {
  Model model;
  REQUIRE(model.getNumReservoirs() == 0);
  REQUIRE(model.getConnectionType() == "serial");

  model.addReservoir(std::make_shared<NvarReservoir>(2), "parallel");
  model.addReservoir(std::make_shared<NvarReservoir>(3), "parallel");
  REQUIRE(model.getNumReservoirs() == 2);
  REQUIRE(model.getConnectionType() == "parallel");
}

namespace {

Eigen::MatrixXd sine(int rows, double phase) {
  Eigen::MatrixXd values(rows, 1);
  for (int i = 0; i < rows; ++i) {
    values(i, 0) = std::sin(0.3 * i + phase);
  }
  return values;
}

// An independent copy with bit-identical parameters and states. Constructing a second
// reservoir from the same seed is not enough: power iteration draws from std::rand().
Model cloneModel(const Model &model) {
  std::stringstream buffer;
  model.save(buffer);
  return Model::load(buffer);
}

// A model trained to predict the next value of a sine wave, so its outputs can be fed back.
Model makeGenerativeModel(int topology, int output_columns = 1) {
  Model model;
  if (topology == 0) {
    model.addReservoir(std::make_shared<RandomSparseReservoir>(30, 0.9, 0.3, 0.5, 1.0, true, 7));
  } else if (topology == 1) {
    model.addReservoir(std::make_shared<RandomSparseReservoir>(8, 0.9, 0.5, 0.5, 1.0, true, 7));
    model.addReservoir(std::make_shared<NvarReservoir>(2, 2));
  } else {
    model.addReservoir(std::make_shared<RandomSparseReservoir>(30, 0.9, 0.3, 0.5, 1.0, true, 7), "parallel");
    model.addReservoir(std::make_shared<NvarReservoir>(2, 2), "parallel");
  }
  model.setReadout(std::make_shared<RidgeReadout>(1e-4));
  const Eigen::MatrixXd series = sine(81, 0.0);
  model.fit(series.topRows(80), series.bottomRows(80).replicate(1, output_columns), 10);
  return model;
}

bool sameStates(const Model &a, const Model &b) {
  for (size_t i = 0; i < a.getNumReservoirs(); ++i) {
    if (a.getReservoir(i)->getState() != b.getReservoir(i)->getState()) {
      return false;
    }
  }
  return true;
}

} // namespace

TEST_CASE("Model - generative prediction in chunks equals one call", "[Model]") {
  const int topology = GENERATE(0, 1, 2); // serial, serial RandomSparse -> NVAR, parallel
  const auto chunks = GENERATE(std::vector<int>{3, 4}, std::vector<int>{6, 1}, std::vector<int>{1, 1, 1, 1, 1, 1, 1});
  const Model trained = makeGenerativeModel(topology);
  const Eigen::MatrixXd prime = sine(10, 2.0);

  Model whole = cloneModel(trained);
  const Eigen::MatrixXd expected = whole.predictGenerative(prime, 7);

  // The first chunk is primed; later chunks continue from the reservoir states.
  Model chunked = cloneModel(trained);
  Eigen::MatrixXd generated(0, 1);
  for (size_t i = 0; i < chunks.size(); ++i) {
    const Eigen::MatrixXd part = chunked.predictGenerative(i == 0 ? prime : Eigen::MatrixXd(0, 1), chunks[i]);
    generated.conservativeResize(generated.rows() + part.rows(), 1);
    generated.bottomRows(part.rows()) = part;
  }
  REQUIRE(generated == expected);
  REQUIRE(sameStates(chunked, whole));
}

TEST_CASE("Model - one-step generation advances the reservoirs", "[Model]") {
  Model model = makeGenerativeModel(0);
  model.predictGenerative(sine(10, 2.0), 1);
  const Eigen::MatrixXd state_before = model.getReservoir(0)->getState();
  const Eigen::MatrixXd first = model.predictGenerative(Eigen::MatrixXd(0, 1), 1);
  REQUIRE(model.getReservoir(0)->getState() != state_before);
  // The next one-step call does not repeat the previous output.
  REQUIRE(model.predictGenerative(Eigen::MatrixXd(0, 1), 1) != first);
}

TEST_CASE("Model - zero-step generation feeds nothing back", "[Model]") {
  const Model trained = makeGenerativeModel(2);
  const Eigen::MatrixXd prime = sine(10, 2.0);

  SECTION("Priming data is still consumed") {
    Model generative = cloneModel(trained);
    Model online = cloneModel(trained);
    REQUIRE(generative.predictGenerative(prime, 0).rows() == 0);
    online.predictOnline(prime); // advances through the same inputs without feedback
    REQUIRE(sameStates(generative, online));
  }
  SECTION("Without priming data the states are unchanged") {
    Model generative = cloneModel(trained);
    REQUIRE(generative.predictGenerative(Eigen::MatrixXd(0, 1), 0).rows() == 0);
    REQUIRE(sameStates(generative, trained));
  }
}

TEST_CASE("Model - generated outputs that cannot be fed back throw", "[Model]") {
  // Two output columns cannot be fed back into reservoirs locked to one input column.
  const int topology = GENERATE(0, 2);
  const int n_steps = GENERATE(1, 3);
  Model model = makeGenerativeModel(topology, 2);
  REQUIRE_THROWS_AS(model.predictGenerative(sine(10, 2.0), n_steps), std::invalid_argument);
  REQUIRE(model.predictGenerative(sine(10, 2.0), 0).cols() == 2);
}
